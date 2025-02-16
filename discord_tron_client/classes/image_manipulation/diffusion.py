from diffusers import models
try:
    from diffusers.loaders import lora_base
except:
    pass
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImageVariationPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusion3Pipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionKDiffusionPipeline,
    DiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    UniPCMultistepScheduler,
    AutoPipelineForText2Image,
    KandinskyV22CombinedPipeline,
    SanaPipeline,
    AuraFlowPipeline,
    LTXPipeline,
    LTXImageToVideoPipeline,
    FluxPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from discord_tron_client.classes.image_manipulation.pipeline_runners.overrides.pixart import (
    PixArtSigmaPipeline,
)
from diffusers import DiffusionPipeline as Pipeline
from typing import Dict
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.app_config import AppConfig
# from discord_tron_client.classes.image_manipulation.face_upscale import (
#     get_upscaler,
#     use_upscaler,
# )
from PIL import Image
import torch, gc, logging, diffusers, transformers, os, time, psutil
from torch import OutOfMemoryError

logger = logging.getLogger("DiffusionPipelineManager")
logger.setLevel("DEBUG")
if not torch.backends.mps.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
    if torch.backends.cuda.mem_efficient_sdp_enabled():
        logger.info("CUDA SDP (scaled dot product attention) is enabled.")
    if torch.backends.cuda.math_sdp_enabled():
        logger.info("CUDA MATH SDP (scaled dot product attention) is enabled.")
else:
    logger.info("MPS is enabled.")

hardware = HardwareInfo()
config = AppConfig()

def pin_pipeline_memory(pipe: diffusers.DiffusionPipeline):
    """
    Recursively pins (page-locks) the .data of all parameters and buffers
    for every torch.nn.Module in the pipeline's components.

    NOTE:
        - Only relevant if the pipeline is on CPU.
        - Typically used for faster CPU→GPU transfers if you plan to
          move back and forth.
        - May increase overall system memory usage because pinned memory 
          cannot be paged out.
    """
    for component_name, component in pipe.components.items():
        if isinstance(component, torch.nn.Module):
            for param in component.parameters():
                # Pin parameter data
                param.data = param.data.pin_memory()
                # If gradients exist and you expect to move them, you could also pin those:
                if param.grad is not None:
                    param.grad.data = param.grad.data.pin_memory()

            for buffer_name, buffer in component.named_buffers():
                if buffer is not None and buffer.device.type == "cpu":
                    buffer.data = buffer.data.pin_memory()

class PipelineRecord:
    """
    Stores a Pipeline along with metadata for LRU/offload management.
    """
    def __init__(self, pipeline: Pipeline, model_id: str, location: str):
        self.pipeline = pipeline
        self.model_id = model_id
        # "cuda", "cpu", or "meta" (fully removed from memory).
        self.location = location
        # For LRU or usage-based heuristics:
        self.last_access_time = time.time()
        self.usage_count = 0

    def update_access(self):
        self.last_access_time = time.time()
        self.usage_count += 1


class DiffusionPipelineManager:
    PIPELINE_CLASSES = {
        "text2img": DiffusionPipeline,
        "sana": SanaPipeline,
        "pixart": PixArtSigmaPipeline,
        "kandinsky-2.2": KandinskyV22CombinedPipeline,
        "prompt_variation": LTXImageToVideoPipeline,
        "variation": StableDiffusionPipeline,
        "upscaler": StableDiffusionPipeline,
    }
    SCHEDULER_MAPPINGS = {
        "DPMSolverMultistepScheduler": diffusers.DPMSolverMultistepScheduler,
        "PNDMScheduler": diffusers.PNDMScheduler,
        "EulerAncestralDiscreteScheduler": diffusers.EulerAncestralDiscreteScheduler,
        "KDPM2AncestralDiscreteScheduler": diffusers.KDPM2AncestralDiscreteScheduler,
        "DDIMScheduler": diffusers.DDIMScheduler,
        "EulerDiscreteScheduler": diffusers.EulerDiscreteScheduler,
        "KDPM2DiscreteScheduler": diffusers.KDPM2DiscreteScheduler,
        "IPNDMScheduler": diffusers.IPNDMScheduler,
        "KarrasVeScheduler": diffusers.KarrasVeScheduler,
    }

    def __init__(self):
        hw_limits = hardware.get_hardware_limits()
        self.torch_dtype = torch.bfloat16
        self.is_memory_constrained = False
        self.model_id = None

        if (
            hw_limits["gpu"] != "Unknown"
            and hw_limits["gpu"] >= 16
            and config.get_precision_bits() == 32
        ):
            self.torch_dtype = torch.float32
        if hw_limits["gpu"] != "Unknown" and hw_limits["gpu"] <= 16:
            logger.warning(
                f"Our GPU has less than 16GB of memory, so we will use memory constrained pipeline parameters for image generation, resulting in higher CPU use to lower VRAM use."
            )
            self.is_memory_constrained = True

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Track concurrency
        self.max_gpu_pipelines = hardware.get_concurrent_pipe_count()
        # Track system CPU memory usage threshold: once exceeded, fully delete older CPU-located pipelines
        self.max_cpu_mem = hardware.get_memory_total() * 0.75
        self.cpu_mem_threshold = 0.75

        # We'll store PipelineRecords in self.pipelines instead of raw Pipeline objects
        self.pipelines: Dict[str, PipelineRecord] = {}

        # Additional tracking as in your original code
        self.last_pipe_type: Dict[str, str] = {}
        self.last_pipe_scheduler: Dict[str, str] = {}
        self.pipeline_versions = {}

        # If your code references this somewhere else:
        self.pipeline_runner = {"model": None}

    def _get_current_cpu_mem_usage(self) -> int:
        """
        Return percentage CPU used memory by this process.
        """
        try:
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss
            logger.info(f"Memory information: {mem}")
            return mem
        except Exception as e:
            logger.error(f"Error getting CPU memory usage: {e}")
            return 0

    def _move_pipeline_to_device(self, record: PipelineRecord, device: str):
        """
        Moves a pipeline to device: "cuda", "cpu", or "meta".
        """
        if record.location == device:
            return  # Already there
        try:
            record.pipeline.to(device, non_blocking=True if device == "cpu" else False)
            record.location = device
        except Exception as e:
            logger.error(f"Error moving pipeline {record.model_id} to {device}: {e}")

    def _offload_one_pipeline_from_gpu(self, exclude_model_id: str = None):
        """
        If GPU concurrency is at max, pick the least recently used pipeline on GPU
        (excluding `exclude_model_id`) and move it to CPU.
        """
        candidates = [
            r for r in self.pipelines.values()
            if r.location == "cuda" and r.model_id != exclude_model_id
        ]
        if not candidates:
            return  # nothing we can offload

        # LRU: sort by last_access_time ascending
        candidates.sort(key=lambda x: x.last_access_time)
        oldest_record = candidates[0]
        logger.info(
            f"Offloading pipeline {oldest_record.model_id} from GPU to CPU to free up concurrency."
        )
        self._move_pipeline_to_device(oldest_record, "cpu")

    def _remove_pipeline_from_memory(self, model_id: str):
        """
        Moves pipeline to "meta" and removes it from self.pipelines entirely.
        """
        if model_id not in self.pipelines:
            return
        record = self.pipelines[model_id]

        # If it's on GPU, move to CPU first (optional step to ensure a clean move):
        if record.location == "cuda":
            self._move_pipeline_to_device(record, "cpu")

        logger.info(f"Fully removing pipeline {model_id} from memory.")
        try:
            record.pipeline.to("meta")
        except Exception as e:
            logger.error(f"Error when moving {model_id} to meta device: {e}")

        # Dereference to help garbage collection
        del record.pipeline
        del self.pipelines[model_id]

        if self.pipeline_runner.get("model") == model_id:
            self.pipeline_runner["model"] = None

        # Clear CUDA cache after removal
        self.clear_cuda_cache()

    def _cleanup_cpu_memory_if_needed(self):
        """
        If CPU usage is above threshold, remove CPU-located pipelines in LRU order
        until we fall below the threshold.
        """
        current_cpu_usage = hardware.get_memory_total() - hardware.get_memory_free()
        limit = int(self.max_cpu_mem * self.cpu_mem_threshold)
        if current_cpu_usage <= limit:
            return

        # We are above CPU memory threshold -> remove some CPU-resident pipelines
        logger.warning(
            f"CPU memory usage {current_cpu_usage} exceeds threshold {limit}. "
            f"Removing older pipelines from CPU memory..."
        )

        # Find all pipelines located on CPU
        candidates = [r for r in self.pipelines.values() if r.location == "cpu"]
        # Sort by last_access_time ascending (LRU)
        candidates.sort(key=lambda x: x.last_access_time)

        idx = 0
        while current_cpu_usage > limit and idx < len(candidates):
            oldest = candidates[idx]
            idx += 1
            self._remove_pipeline_from_memory(oldest.model_id)
            current_cpu_usage = self._get_current_cpu_mem_usage()

    def _ensure_pipeline_on_gpu(self, model_id: str):
        """
        Moves the pipeline with model_id to the GPU if possible; 
        if concurrency is exceeded, offload an LRU pipeline first.
        """
        if model_id not in self.pipelines:
            return

        record = self.pipelines[model_id]
        if record.location == "cuda":
            return  # already on GPU

        # If we have room on the GPU, just move it:
        if self.num_pipelines_on_gpu() < self.max_gpu_pipelines:
            self._move_pipeline_to_device(record, "cuda")
        else:
            # Offload an LRU pipeline first
            self._offload_one_pipeline_from_gpu(exclude_model_id=model_id)
            self._move_pipeline_to_device(record, "cuda")

    def num_pipelines_on_gpu(self) -> int:
        return sum(1 for r in self.pipelines.values() if r.location == "cuda")

    def clear_pipeline(self, model_id: str) -> None:
        """
        Overridden to fully remove pipeline from memory rather than just from GPU.
        (Interface remains intact.)
        """
        if model_id in self.pipelines:
            self._remove_pipeline_from_memory(model_id)
        else:
            logger.warning(f"Model {model_id} did not have a cached pipeline to clear.")

    def create_pipeline(
        self,
        model_id: str,
        pipe_type: str,
        use_safetensors: bool = True,
        custom_text_encoder=None,
        safety_modules: dict = None,
    ) -> Pipeline:
        pipeline_class = self.PIPELINE_CLASSES[pipe_type]
        if "pixart" in model_id:
            pipeline_class = self.PIPELINE_CLASSES["pixart"]
        if "sana" in model_id.lower():
            print("Using Sana pipeline class.")
            pipeline_class = self.PIPELINE_CLASSES["sana"]

        extra_args = {
            "feature_extractor": None,
            "safety_checker": None,
            "requires_safety_checker": None,
        }
        if safety_modules is not None:
            for key in safety_modules:
                extra_args[key] = safety_modules[key]

        if pipe_type in ["variation", "upscaler"]:
            logger.debug(f"Creating a ControlNet model for {model_id}")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=self.torch_dtype
            )
            logger.debug(
                f"Passing the ControlNet into a StableDiffusionControlNetPipeline for {model_id}"
            )
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                custom_pipeline="stable_diffusion_controlnet_img2img",
                controlnet=controlnet,
                use_safetensors=use_safetensors,
                **extra_args,
            )
        elif pipe_type in ["prompt_variation"]:
            logger.debug(f"Creating a prompt_variation pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=use_safetensors,
                **extra_args,
            )
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
        elif pipe_type in ["text2img"]:
            logger.debug(f"Creating a txt2img pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=use_safetensors,
                use_auth_token=config.get_huggingface_api_key(),
                variant=config.get_config_value("model_default_variant", None),
                **extra_args,
            )
            logger.debug(f"Model config: {pipeline.config}")
        else:
            logger.debug(f"Using standard pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=use_safetensors,
                use_auth_token=config.get_huggingface_api_key(),
                **extra_args,
            )
        quanto_quantized_models = [
            LTXPipeline, LTXImageToVideoPipeline, FluxPipeline
        ]
        if type(pipeline) in quanto_quantized_models and not hasattr(pipeline, "quantized"):
            logger.info(f"Quantizing the model for {model_id}")

            from optimum.quanto import quantize, freeze, qint8
            quantize(pipeline.transformer, weights=qint8, include=[
                "*transformer*",
            ])
            logger.info(f"Freezing the model for {model_id}")
            freeze(pipeline.transformer)
            self.delete_pipes(keep_model=model_id)
            # pipeline.to(self.device)

            # from torchao.quantization import quantize_, int8_weight_only, autoquant
            # # pipeline.transformer.to(memory_format=torch.channels_last)
            # pipeline.transformer = torch.compile(
            #     pipeline.transformer, mode="reduce-overhead", fullgraph=True
            # )
            # # quantize_(pipeline.transformer, int8_weight_only(), device="cuda")
            # pipeline.transformer = autoquant(pipeline.transformer, device="cuda")

            setattr(pipeline, "quantized", True)
        # Disable safety checker
        if hasattr(pipeline, "safety_checker") and pipeline.safety_checker is not None:
            pipeline.safety_checker = lambda images, clip_input: (images, False)
        if hasattr(pipeline, "watermark") and pipeline.watermark is not None:
            pipeline.watermark = None
        if hasattr(pipeline, "watermarker") and pipeline.watermarker is not None:
            pipeline.watermarker = None

        # Move pipeline to CPU initially
        pin_pipeline_memory(pipe=pipeline)
        return pipeline

    def upscale_image(self, image: Image):
        return image
        # self._initialize_upscaler_pipe()
        # esrgan_upscaled = use_upscaler(self.pipelines["upscaler"], image)
        # return esrgan_upscaled

    def get_model_latest_hash(
        self,
        model_id: str,
        subfolder: str = "unet",
        unet_model_name: str = "diffusion_pytorch_model.safetensors",
    ) -> str:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        try:
            url = hf_hub_url(
                repo_id=model_id, filename=os.path.join(subfolder, unet_model_name)
            )
            logger.debug(f"Retrieving metadata from URL: {url}")
            metadata = get_hf_file_metadata(url)
            result = metadata.commit_hash
            logger.debug(f"Commit hash retrieved: {result}")
            return result
        except Exception as e:
            url = hf_hub_url(
                repo_id=model_id, filename=os.path.join("transformer", unet_model_name)
            )
            logger.error(f"Could not get model metadata: {e}")
            try:
                metadata = get_hf_file_metadata(url)
                result = metadata.commit_hash
                logger.debug(f"Commit hash retrieved: {result}")
                return result
            except Exception as e:
                logger.error(f"Could not get model metadata: {e}")
                return False

    def get_repo_last_modified(self, model_id: str) -> str:
        from huggingface_hub import model_info
        model_info_obj = model_info(model_id)
        last_modified = str(model_info_obj.last_modified).split("+")[0]
        return last_modified

    def is_model_latest(self, model_id: str) -> bool:
        latest_hash = self.get_model_latest_hash(model_id)
        if latest_hash is None:
            logger.debug(f"is_model_latest could not retrieve metadata: {latest_hash}")
            return None
        if latest_hash is False:
            logger.debug(
                f"is_model_latest could not retrieve metadata: {latest_hash}, but we are assuming it's fine."
            )
            return True

        current_hash = self.pipeline_versions.get(model_id, {}).get("latest_hash", "unknown")
        last_modified = self.pipeline_versions.get(model_id, {}).get("last_modified", "unknown")
        latest_modified = self.get_repo_last_modified(model_id)
        test = (latest_hash == current_hash) and (last_modified == latest_modified)
        if test:
            logger.debug(f"Model {model_id} is the latest version, modified on {last_modified}.")
            return True

        logger.debug(
            f"Model {model_id} is not the latest. Setting version from {current_hash} to {latest_hash}"
        )
        self.pipeline_versions[model_id] = {
            "latest_hash": latest_hash,
            "last_modified": latest_modified,
        }
        return False

    def get_pipe(
        self,
        user_config: dict,
        model_id: str,
        prompt_variation: bool = False,
        promptless_variation: bool = False,
        upscaler: bool = False,
        custom_text_encoder=None,
        safety_modules: dict = None,
        use_safetensors: bool = True,
    ) -> Pipeline:
        """
        Main retrieval function, now updated to offload pipelines to CPU
        rather than deleting them. Also calls CPU memory cleanup when done.
        """
        pipe_type = (
            "prompt_variation"
            if prompt_variation
            else "variation"
            if promptless_variation
            else "upscaler"
            if upscaler
            else "text2img"
        )
        if "kandinsky-2-2" in model_id:
            use_safetensors = False
            pipe_type = "kandinsky-2.2"

        logger.info(
            f"Executing get_pipe for model {model_id} with pipe_type={pipe_type} safetensors={use_safetensors}"
        )

        # Check if model is outdated
        logger.info(
            f"Checking the model version for {model_id}: currently we have {self.pipeline_versions.get(model_id, {}).get('latest_hash', 'unknown')}"
        )
        if not self.is_model_latest(model_id):
            new_revision = self.pipeline_versions.get(model_id, {}).get("latest_hash", None)
            if not new_revision:
                raise ValueError(f"Could not get the latest revision for model {model_id}")
            logger.warning(
                f"Model {model_id} is not the latest version. Deleting the stored model. Retrieving {new_revision} from the cache."
            )
            self.clear_pipeline(model_id)

        # If pipeline doesn't exist, create it on CPU
        if model_id not in self.pipelines:
            logger.debug(
                f"Creating pipeline type {pipe_type} for model {model_id} with custom_text_encoder {type(custom_text_encoder)}"
            )
            new_pipeline = self.create_pipeline(
                model_id,
                pipe_type,
                use_safetensors=use_safetensors,
                custom_text_encoder=custom_text_encoder,
                safety_modules=safety_modules,
            )
            self.pipelines[model_id] = PipelineRecord(new_pipeline, model_id, location="cpu")
            self.last_pipe_type[model_id] = pipe_type
        else:
            logger.info(f"Using existing pipeline for {model_id}. Checking concurrency constraints.")

        # Move pipeline to GPU if needed, offloading something else if concurrency is maxed
        self._ensure_pipeline_on_gpu(model_id)

        # Update usage stats
        record = self.pipelines[model_id]
        record.update_access()

        # Additional VAE tiling logic from your code
        enable_tiling = user_config.get("enable_tiling", True)
        if hasattr(record.pipeline, "vae") and enable_tiling:
            logger.warning(f"Enabling VAE tiling. This could cause artifacted outputs.")
            record.pipeline.vae.enable_tiling()
            record.pipeline.vae.enable_slicing()

        # Return pipeline after concurrency management
        # Now that we've loaded the pipeline on GPU, let's do CPU memory cleanup if needed
        self._cleanup_cpu_memory_if_needed()

        return record.pipeline

    def delete_pipes(self, keep_model: str = None):
        """
        Previously, this method would forcibly remove all pipes except keep_model.
        Now we update it to offload pipelines from GPU if concurrency is exceeded
        and remove from CPU entirely only if needed (LRU).
        We'll interpret "delete_pipes" as "try to reduce GPU usage to keep only keep_model on GPU".
        """
        for model_id, record in list(self.pipelines.items()):
            if keep_model is not None and model_id == keep_model:
                continue
            # If it's on GPU, move to CPU to free concurrency slots
            if record.location == "cuda":
                logger.info(f"Offloading pipeline {model_id} from GPU to CPU (delete_pipes).")
                self._move_pipeline_to_device(record, "cpu")

        # Optionally do a CPU memory cleanup if needed
        self._cleanup_cpu_memory_if_needed()

    def clear_cuda_cache(self):
        """
        Clears Python garbage, plus optionally empties PyTorch CUDA cache
        if configured. 
        """
        gc.collect()
        if config.get_cuda_cache_clear_toggle():
            logger.info("Clearing the CUDA cache...")
            torch.cuda.empty_cache()
            torch.clear_autocast_cache()
        else:
            logger.debug(
                f"NOT clearing CUDA cache. Config option `cuda_cache_clear` is not set, or is False."
            )

    def get_controlnet_pipe(self):
        self.delete_pipes()
        pipeline = self.get_pipe(
            promptless_variation=True,
            user_config={},
            model_id="emilianJR/epiCRealism",
            use_safetensors=False,
        )
        return pipeline

    def get_sdxl_refiner_pipe(self):
        refiner_model = config.get_config_value(
            "refiner_model", "stabilityai/stable-diffusion-xl-refiner-1.0"
        )
        self.delete_pipes(keep_model=refiner_model)
        pipeline = self.get_pipe(
            user_config={}, model_id=refiner_model, prompt_variation=True
        )
        pipeline.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            use_auth_token=config.get_huggingface_api_key(),
        ).to(self.device)
        return pipeline
