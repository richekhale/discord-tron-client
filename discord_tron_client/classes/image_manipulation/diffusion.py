import tracemalloc

tracemalloc.start()
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
from PIL import Image
import torch, gc, logging, diffusers, transformers, os, time, psutil
from torch import OutOfMemoryError
import json

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
    """
    for component_name, component in pipe.components.items():
        if isinstance(component, torch.nn.Module):
            for param in component.parameters():
                param.data = param.data.pin_memory()
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
        # For usage-based heuristics:
        self.last_access_time = time.time()
        self.usage_count = 0
        # Track creation time so older, less-used pipelines can be offloaded first
        self.creation_time = time.time()

    def update_access(self):
        self.last_access_time = time.time()
        self.usage_count += 1

    def get_offload_score(self) -> float:
        """
        Combine multiple factors to determine which pipeline to offload first:
          - If a pipeline is old and not accessed recently, its score gets bigger.
          - A pipeline that was used a lot gets a negative offset so it stays around longer.
        """
        now = time.time()
        time_since_access = now - self.last_access_time
        pipeline_age = now - self.creation_time

        # Tweak the weights as you like:
        # Larger usage_count => keep pipeline => subtract from score
        # Larger time_since_access => bigger score => more likely to offload
        # Larger pipeline_age => bigger score => more likely to offload
        score = (time_since_access) + 0.5 * (pipeline_age) - 50 * (self.usage_count)
        return score


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
                "Our GPU has less than 16GB of memory, so we will use memory-constrained pipeline parameters."
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
        # Track CPU memory usage threshold
        self.max_cpu_mem = hardware.get_memory_total() - 48
        self.cpu_mem_threshold = 0.75

        # We'll store PipelineRecords in self.pipelines
        self.pipelines: Dict[str, PipelineRecord] = {}

        self.last_pipe_type: Dict[str, str] = {}
        self.last_pipe_scheduler: Dict[str, str] = {}
        self.pipeline_versions = {}
        self.pipeline_runner = {"model": None}

        self.vram_usage_map = {}
        self._load_vram_usage_cache()

    def _load_vram_usage_cache(self):
        cache_path = "vram_usage_cache.json"
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self.vram_usage_map = json.load(f)
                for k, v in list(self.vram_usage_map.items()):
                    if not isinstance(v, int) and not isinstance(v, float):
                        self.vram_usage_map[k] = float(v)
            except Exception as e:
                logger.error(f"Error loading VRAM usage cache: {e}")

    def _save_vram_usage_cache(self):
        cache_path = "vram_usage_cache.json"
        try:
            with open(cache_path, "w") as f:
                json.dump(self.vram_usage_map, f)
        except Exception as e:
            logger.error(f"Error saving VRAM usage cache: {e}")

    def _get_current_cpu_mem_usage(self) -> int:
        """
        Return a rough measure of system memory usage
        """
        usage = 0
        try:
            # Custom logic for CPU usage
            for model, record in self.pipelines.items():
                if record.location == "cpu" and model in self.vram_usage_map:
                    usage_multiplier = 1.5
                    usage += self.vram_usage_map[model] * usage_multiplier
        except Exception as e:
            logger.error(f"Error getting CPU memory usage: {e}")
        return usage

    def _move_pipeline_to_device(self, record: PipelineRecord, device: str):
        """
        Move a pipeline to a new device, updating the record's location.

        Args:
            record (PipelineRecord): The pipeline record object.
            device (str): The device to move the pipeline to ("cuda", "cpu", or "meta").
        """
        if record.location == device:
            return
        try:
            if device == "cuda" and torch.cuda.is_available():
                # If we haven't measured VRAM usage yet, do so
                if (
                    record.model_id not in self.vram_usage_map
                    or self.vram_usage_map[record.model_id] == 0
                ):
                    mem_before = torch.cuda.memory_allocated()
                    record.pipeline.to(device, non_blocking=False)
                    mem_after = torch.cuda.memory_allocated()
                    used_bytes = mem_after - mem_before
                    used_gb = used_bytes / 2**30
                    if used_gb > 0:
                        self.vram_usage_map[record.model_id] = used_gb
                        logger.info(
                            f"Pipeline {record.model_id} uses ~{used_gb:.2f} GB VRAM (measured)."
                        )
                        self._save_vram_usage_cache()
                    else:
                        logger.info(
                            f"Measured VRAM usage for {record.model_id} = {used_gb} GB, skipping update."
                        )
                else:
                    record.pipeline.to(device, non_blocking=False)
                    cached_gb = self.vram_usage_map[record.model_id]
                    logger.info(
                        f"Pipeline {record.model_id} VRAM usage is ~{cached_gb:.2f} GB (cached)."
                    )
            else:
                # Move to CPU or meta
                record.pipeline.to(device, non_blocking=(device == "cpu"))
            record.location = device
        except Exception as e:
            logger.error(f"Error moving pipeline {record.model_id} to {device}: {e}")

    def _offload_one_pipeline_from_gpu(self, exclude_model_id: str = None):
        """
        If GPU concurrency is at max, pick the pipeline with the highest "offload_score"
        (meaning it's the best candidate to remove), excluding exclude_model_id.
        """
        candidates = [
            r
            for r in self.pipelines.values()
            if r.location == "cuda" and r.model_id != exclude_model_id
        ]
        if not candidates:
            return

        # Sort by get_offload_score() descending => highest score offloaded first
        candidates.sort(key=lambda x: x.get_offload_score(), reverse=True)
        worst_record = candidates[0]
        logger.info(
            f"Offloading pipeline {worst_record.model_id} from GPU to CPU (score={worst_record.get_offload_score():.2f})."
        )
        self._move_pipeline_to_device(worst_record, "cpu")

    def _remove_pipeline_from_memory(self, model_id: str):
        if model_id not in self.pipelines:
            return
        record = self.pipelines[model_id]

        if record.location == "cuda":
            self._move_pipeline_to_device(record, "cpu")

        logger.info(f"Fully removing pipeline {model_id} from memory.")
        try:
            record.pipeline.to("meta")
        except Exception as e:
            logger.error(f"Error moving {model_id} to meta: {e}")

        del record.pipeline
        del self.pipelines[model_id]

        if self.pipeline_runner.get("model") == model_id:
            self.pipeline_runner["model"] = None

        self.clear_cuda_cache()

    def _cleanup_cpu_memory_if_needed(self):
        """
        If CPU memory usage exceeds the threshold, remove the oldest pipelines.
        """
        current_cpu_usage = self._get_current_cpu_mem_usage()
        if current_cpu_usage <= self.max_cpu_mem:
            logger.info(
                f"Not clearing memory, usage {current_cpu_usage} < {self.max_cpu_mem}"
            )
            return

        logger.warning(
            f"CPU memory usage {current_cpu_usage} exceeds threshold {self.max_cpu_mem}. "
            f"Removing older pipelines from CPU memory..."
        )

        # Collect pipelines on CPU, sort by offload_score descending
        candidates = [r for r in self.pipelines.values() if r.location == "cpu"]
        candidates.sort(key=lambda x: x.get_offload_score(), reverse=True)

        idx = 0
        while current_cpu_usage > self.max_cpu_mem and idx < len(candidates):
            worst_record = candidates[idx]
            idx += 1
            self._remove_pipeline_from_memory(worst_record.model_id)
            current_cpu_usage = self._get_current_cpu_mem_usage()
            self.clear_cuda_cache()
            logger.info(f"New memory usage level {current_cpu_usage}")

    def _ensure_pipeline_on_gpu(self, model_id: str):
        """
        If the pipeline is not on the GPU, move it there if possible.

        Args:
            model_id (str): The model ID of the pipeline to check.
        """
        if model_id not in self.pipelines:
            return
        record = self.pipelines[model_id]
        if record.location == "cuda":
            return
        if self.num_pipelines_on_gpu() < self.max_gpu_pipelines:
            self._move_pipeline_to_device(record, "cuda")
        else:
            self._offload_one_pipeline_from_gpu(exclude_model_id=model_id)
            self._move_pipeline_to_device(record, "cuda")

    def num_pipelines_on_gpu(self) -> int:
        """
        Return the number of pipelines currently on the GPU.

        Returns:
            int: The number of pipelines on the GPU.
        """
        return sum(1 for r in self.pipelines.values() if r.location == "cuda")

    def clear_pipeline(self, model_id: str) -> None:
        """
        Clear a pipeline from memory, if it exists.

        Args:
            model_id (str): The model ID of the pipeline to clear.
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
        """
        Create a new pipeline of the specified type.

        Args:
            model_id (str): The model ID to create a pipeline for.
            pipe_type (str): The type of pipeline to create.
            use_safetensors (bool, optional): Whether to use safetensors only. Defaults to True.
            custom_text_encoder (_type_, optional): Whether the pipeline comes with a custom text encoder. Defaults to None.
            safety_modules (dict, optional): Any additional safety modules. Defaults to None.

        Returns:
            Pipeline: The created pipeline.
        """
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
            logger.debug(f"StableDiffusionControlNetPipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                custom_pipeline="stable_diffusion_controlnet_img2img",
                controlnet=controlnet,
                use_safetensors=use_safetensors,
                **extra_args,
            )
        elif pipe_type == "prompt_variation":
            logger.debug(f"Creating a prompt_variation pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=use_safetensors,
                **extra_args,
            )
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
        elif pipe_type == "text2img":
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

        quanto_quantized_models = [LTXPipeline, LTXImageToVideoPipeline, FluxPipeline]
        if type(pipeline) in quanto_quantized_models and not hasattr(
            pipeline, "quantized"
        ):
            logger.info(f"Quantizing the model for {model_id}")
            from optimum.quanto import quantize, freeze, qint8

            quantize(pipeline.transformer, weights=qint8, include=["*transformer*"])
            logger.info(f"Freezing the model for {model_id}")
            freeze(pipeline.transformer)
            self.delete_pipes(keep_model=model_id)
            setattr(pipeline, "quantized", True)

        if hasattr(pipeline, "safety_checker") and pipeline.safety_checker is not None:
            pipeline.safety_checker = lambda images, clip_input: (images, False)
        if hasattr(pipeline, "watermark") and pipeline.watermark is not None:
            pipeline.watermark = None
        if hasattr(pipeline, "watermarker") and pipeline.watermarker is not None:
            pipeline.watermarker = None

        pin_pipeline_memory(pipe=pipeline)
        return pipeline

    def upscale_image(self, image: Image):
        """
        No-op

        Args:
            image (Image)

        Returns:
            Input image
        """
        return image

    def get_model_latest_hash(
        self,
        model_id: str,
        subfolder: str = "unet",
        unet_model_name: str = "diffusion_pytorch_model.safetensors",
    ) -> str:
        """
        Get the latest commit hash for a model.

        Currently disabled.

        Args:
            model_id (str)
            subfolder (str, optional): Defaults to "unet".
            unet_model_name (str, optional): Defaults to "diffusion_pytorch_model.safetensors".

        Returns:
            str
        """
        return None
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        try:
            url = hf_hub_url(
                repo_id=model_id, filename=os.path.join(subfolder, unet_model_name)
            )
            metadata = get_hf_file_metadata(url)
            return metadata.commit_hash
        except Exception as e:
            url = hf_hub_url(
                repo_id=model_id, filename=os.path.join("transformer", unet_model_name)
            )
            logger.error(f"Could not get model metadata: {e}")
            try:
                metadata = get_hf_file_metadata(url)
                return metadata.commit_hash
            except Exception as e:
                logger.error(f"Could not get model metadata: {e}")
                return False

    def get_repo_last_modified(self, model_id: str) -> str:
        return None
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
                f"is_model_latest could not retrieve metadata: {latest_hash}, but assuming it's fine."
            )
            return True

        current_hash = self.pipeline_versions.get(model_id, {}).get(
            "latest_hash", "unknown"
        )
        last_modified = self.pipeline_versions.get(model_id, {}).get(
            "last_modified", "unknown"
        )
        latest_modified = self.get_repo_last_modified(model_id)
        if (latest_hash == current_hash) and (last_modified == latest_modified):
            logger.debug(
                f"Model {model_id} is the latest version, modified on {last_modified}."
            )
            return True

        logger.debug(
            f"Model {model_id} not latest. Updating hash from {current_hash} to {latest_hash}"
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
            f"get_pipe: {model_id}, type={pipe_type}, safetensors={use_safetensors}"
        )
        logger.info(f"Checking version for {model_id}")

        if not self.is_model_latest(model_id):
            new_revision = self.pipeline_versions.get(model_id, {}).get(
                "latest_hash", None
            )
            if not new_revision:
                raise ValueError(
                    f"Could not get the latest revision for model {model_id}"
                )
            logger.warning(
                f"Model {model_id} is not latest. Clearing and reloading revision {new_revision}."
            )
            self.clear_pipeline(model_id)

        if model_id not in self.pipelines:
            logger.debug(f"Creating pipeline type {pipe_type} for model {model_id}")
            snapshot1 = tracemalloc.take_snapshot()
            new_pipeline = self.create_pipeline(
                model_id,
                pipe_type,
                use_safetensors=use_safetensors,
                custom_text_encoder=custom_text_encoder,
                safety_modules=safety_modules,
            )
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot1, "lineno")
            logger.info("[ Top 10 differences ]")
            for stat in top_stats[:10]:
                logger.info(stat)

            self.pipelines[model_id] = PipelineRecord(
                new_pipeline, model_id, location="cpu"
            )
            self.last_pipe_type[model_id] = pipe_type
        else:
            logger.info(f"Using existing pipeline for {model_id}.")

        self._ensure_pipeline_on_gpu(model_id)
        record = self.pipelines[model_id]
        record.update_access()

        # If user config says to enable tiling, do so
        enable_tiling = user_config.get("enable_tiling", True)
        if hasattr(record.pipeline, "vae") and enable_tiling:
            logger.warning("Enabling VAE tiling (could cause artifacts).")
            record.pipeline.vae.enable_tiling()
            record.pipeline.vae.enable_slicing()

        self._cleanup_cpu_memory_if_needed()
        return record.pipeline

    def delete_pipes(self, keep_model: str = None):
        """
        Offload from GPU, then do normal CPU cleanup if needed.
        """
        for model_id, record in list(self.pipelines.items()):
            if keep_model is not None and model_id == keep_model:
                continue
            if record.location == "cuda":
                logger.info(f"Offloading pipeline {model_id} to CPU (delete_pipes).")
                self._move_pipeline_to_device(record, "cpu")
        self._cleanup_cpu_memory_if_needed()

    def clear_cuda_cache(self):
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        gc.collect()
        if config.get_cuda_cache_clear_toggle():
            logger.info("Clearing the CUDA cache...")
            torch.cuda.empty_cache()
            torch.clear_autocast_cache()
        else:
            logger.debug("NOT clearing CUDA cache (config disabled).")

    def get_controlnet_pipe(self):
        self.delete_pipes()
        return self.get_pipe(
            user_config={},
            model_id="emilianJR/epiCRealism",
            promptless_variation=True,
            use_safetensors=False,
        )

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
