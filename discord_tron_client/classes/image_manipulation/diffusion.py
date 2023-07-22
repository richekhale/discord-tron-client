from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImageVariationPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionUpscalePipeline,
    StableDiffusionKDiffusionPipeline,
    DiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    UniPCMultistepScheduler,
    KandinskyV22Pipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers import DiffusionPipeline as Pipeline
from typing import Dict
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.image_manipulation.face_upscale import (
    get_upscaler,
    use_upscaler,
)
from PIL import Image
import torch, gc, logging, diffusers

torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
if torch.backends.cuda.mem_efficient_sdp_enabled():
    logging.info("CUDA SDP (scaled dot product attention) is enabled.")
if torch.backends.cuda.math_sdp_enabled():
    logging.info("CUDA MATH SDP (scaled dot product attention) is enabled.")
hardware = HardwareInfo()
config = AppConfig()


class DiffusionPipelineManager:
    PIPELINE_CLASSES = {
        "text2img": DiffusionPipeline,
        "kandinsky-2.2": DiffusionPipeline,
        "prompt_variation": StableDiffusionXLImg2ImgPipeline,
        "variation": StableDiffusionPipeline,
        "upscaler": StableDiffusionPipeline,
    }
    SCHEDULER_MAPPINGS = {
        "DPMSolverMultistepScheduler": diffusers.DPMSolverMultistepScheduler,
        "PNDMScheduler": diffusers.PNDMScheduler,
        "EulerAncestralDiscreteScheduler": diffusers.EulerAncestralDiscreteScheduler,
        "EulerDiscreteScheduler": diffusers.EulerDiscreteScheduler,
        "KDPM2AncestralDiscreteScheduler": diffusers.KDPM2AncestralDiscreteScheduler,
        "DDIMScheduler": diffusers.DDIMScheduler,
        "EulerDiscreteScheduler": diffusers.EulerDiscreteScheduler,
        "KDPM2DiscreteScheduler": diffusers.KDPM2DiscreteScheduler,
        "IPNDMScheduler": diffusers.IPNDMScheduler,
        "KarrasVeScheduler": diffusers.KarrasVeScheduler,
    }

    def __init__(self):
        self.pipelines = {}
        hw_limits = hardware.get_hardware_limits()
        self.torch_dtype = torch.float16
        self.is_memory_constrained = False
        self.model_id = None
        if (
            hw_limits["gpu"] != "Unknown"
            and hw_limits["gpu"] >= 16
            and config.get_precision_bits() == 32
        ):
            self.torch_dtype = torch.float32
        if hw_limits["gpu"] != "Unknown" and hw_limits["gpu"] <= 16:
            logging.warn(
                f"Our GPU has less than 16GB of memory, so we will use memory constrained pipeline parameters for image generation, resulting in much higher CPU use to lower VMEM use."
            )
            self.is_memory_constrained = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_pipe_type = {}  # { "model_id": "text2img", ... }
        self.last_pipe_scheduler = {}  # { "model_id": "default" }
        self.pipelines: Dict[str, Pipeline] = {}
        self.last_pipe_type: Dict[str, str] = {}

    def clear_pipeline(self, model_id: str) -> None:
        if model_id in self.pipelines:
            try:
                del self.pipelines[model_id]
                self.clear_cuda_cache()
            except Exception as e:
                logging.error(f"Error when deleting pipe: {e}")

    def create_pipeline(self, model_id: str, pipe_type: str, use_safetensors: bool = True, custom_text_encoder: int = None) -> Pipeline:
        pipeline_class = self.PIPELINE_CLASSES[pipe_type]
        extra_args = {}
        if custom_text_encoder is not None and custom_text_encoder == -1:
            # Disable text encoder.
            extra_args["text_encoder"] = None
        if pipe_type in ["variation", "upscaler"]:
            # Variation uses ControlNet stuff.
            logging.debug(f"Creating a ControlNet model for {model_id}")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=self.torch_dtype
            )
            logging.debug(
                f"Passing the ControlNet into a StableDiffusionControlNetPipeline for {model_id}"
            )
            pipeline = self.PIPELINE_CLASSES["text2img"].from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                custom_pipeline="stable_diffusion_controlnet_img2img",
                controlnet=controlnet,
                feature_extractor=None,
                safety_checker=None,
                requires_safety_checker=None,
                use_safetensors=use_safetensors,
                **extra_args
            )
        elif pipe_type in ["prompt_variation"]:
            # Use the long prompt weighting pipeline.
            logging.debug(f"Creating a LPW pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                feature_extractor=None,
                safety_checker=None,
                requires_safety_checker=None,
                use_safetensors=use_safetensors,
                **extra_args
            )
        elif pipe_type in ["text2img"]:
            logging.debug(f"Creating a txt2img pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                feature_extractor=None,
                safety_checker=None,
                requires_safety_checker=None,
                use_safetensors=use_safetensors,
                use_auth_token=config.get_huggingface_api_key(),
                **extra_args
            )
            logging.debug(f"Model config: {pipeline.config}")
        else:
            logging.debug(f"Using standard pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id, torch_dtype=self.torch_dtype
                **extra_args
            )
        if hasattr(pipeline, "safety_checker") and pipeline.safety_checker is not None:
            pipeline.safety_checker = lambda images, clip_input: (images, False)
        return pipeline

    def upscale_image(self, image: Image):
        self._initialize_upscaler_pipe()
        def resize_for_condition_image(input_image: Image, resolution: int):
            input_image = input_image.convert("RGB")
            W, H = input_image.size
            k = float(resolution) / min(H, W)
            H *= k
            W *= k
            H = int(round(H / 64.0)) * 64
            W = int(round(W / 64.0)) * 64
            img = input_image.resize((W, H), resample=Image.LANCZOS)
            return img

        esrgan_upscaled = use_upscaler(self.pipelines["upscaler"], image)
        # reasonable_size = resize_for_condition_image(esrgan_upscaled, 2048)
        return esrgan_upscaled

    def _initialize_upscaler_pipe(self):
        if "upscaler" not in self.pipelines:
            self.pipelines["upscaler"] = get_upscaler()
        return self.pipelines["upscaler"]

    def get_pipe(
        self,
        user_config: dict,
        scheduler_config: dict,
        model_id: str,
        prompt_variation: bool = False,
        promptless_variation: bool = False,
        upscaler: bool = False,
        custom_text_encoder: int = None
    ) -> Pipeline:
        self.delete_pipes(keep_model=model_id)
        pipe_type = (
            "prompt_variation"
            if prompt_variation
            else "variation"
            if promptless_variation
            else "upscaler"
            if upscaler
            else "text2img"
        )
        use_safetensors = True
        if "kandinsky-2-2" in model_id:
            use_safetensors = False
            pipe_type = "kandinsky-2.2"
        logging.info(
            f"Executing get_pipe for model {model_id} and pipe_type {pipe_type}"
        )

        if (
            model_id in self.last_pipe_type
            and self.last_pipe_type[model_id] != pipe_type
        ):
            logging.warn(
                f"Clearing out an incorrect pipeline type for the same model. Going from {self.last_pipe_type[model_id]} to {pipe_type}. Model: {model_id}"
            )
            self.clear_pipeline(model_id)
        if (
            scheduler_config is not None
            and scheduler_config != {}
            and model_id in self.last_pipe_scheduler
            and self.last_pipe_scheduler[model_id] != scheduler_config["name"]
        ):
            logging.warn(
                f"Clearing out an incorrect pipeline and scheduler, for the same model. Going from {self.last_pipe_scheduler[model_id]} to {scheduler_config['name']}. Model: {model_id}"
            )
            self.clear_pipeline(model_id)

        if model_id not in self.pipelines:
            logging.debug(f"Creating pipeline type {pipe_type} for model {model_id} with custom_text_encoder {custom_text_encoder}")
            self.pipelines[model_id] = self.create_pipeline(model_id, pipe_type, use_safetensors=use_safetensors, custom_text_encoder=custom_text_encoder)
            if pipe_type in ["upscaler", "prompt_variation", "text2img", "kandinsky-2.2"]:
                pass
            elif pipe_type == "variation":
                # I think this needs a specific scheduler set.
                logging.debug(
                    f"Before setting scheduler: {self.pipelines[model_id].scheduler}"
                )
                self.pipelines[
                    model_id
                ].scheduler = UniPCMultistepScheduler.from_config(
                    self.pipelines[model_id].scheduler.config
                )
                logging.debug(
                    f"After setting scheduler: {self.pipelines[model_id].scheduler}"
                )
            # Additional offload settings that we apply to all pipelines.
            if (
                hasattr(self.pipelines[model_id], "enable_model_cpu_offload")
                and hardware.should_offload()
                and not hardware.should_sequential_offload()
            ):
                try:
                    logging.warn(
                        f"Hardware constraints are enabling model CPU offload. This could impact performance."
                    )
                    self.pipelines[model_id].enable_model_cpu_offload()
                except Exception as e:
                    logging.error(f"Could not enable CPU offload on the model: {e}")
            elif (
                hasattr(self.pipelines[model_id], "enable_sequential_cpu_offload")
                and hardware.should_sequential_offload()
            ):
                try:
                    logging.warn(
                        f"Hardware constraints are enabling *SEQUENTIAL* CPU offload. This WILL impact performance."
                    )
                    self.pipelines[model_id].enable_model_cpu_offload()
                except Exception as e:
                    logging.error(
                        f"Could not enable sequential CPU offload on the model: {e}"
                    )
            else:
                logging.info(
                    f"Moving pipe to CUDA early, because no offloading is being used."
                )
                self.pipelines[model_id].to(self.device)
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.log_level = logging.WARNING
                if config.enable_compile() and hasattr(self.pipelines[model_id], 'unet'):
                    self.pipelines[model_id].unet = torch.compile(
                        self.pipelines[model_id].unet,
                        mode="reduce-overhead",
                        fullgraph=True,
                    )
        else:
            logging.info(f"Keeping existing pipeline. Not creating any new ones.")
            self.pipelines[model_id].to(self.device)
        self.last_pipe_type[model_id] = pipe_type
        if scheduler_config is not None and scheduler_config != {}:
            self.last_pipe_scheduler[model_id] = scheduler_config.get("name", "default")
        enable_tiling = user_config.get("enable_tiling", True)
        if hasattr(self.pipelines[model_id], 'vae') and enable_tiling:
            logging.warn(f"Enabling VAE tiling. This could cause artifacted outputs.")
            self.pipelines[model_id].vae.enable_tiling()
            self.pipelines[model_id].vae.enable_slicing()
        elif hasattr(self.pipelines[model_id], 'vae'):
            self.pipelines[model_id].vae.disable_tiling()
            self.pipelines[model_id].vae.disable_slicing()
        return self.pipelines[model_id]

    def delete_pipes(self, keep_model: str = None):
        total_allowed_concurrent = hardware.get_concurrent_pipe_count()
        # Loop by a range of 0 through len(self.pipelines):
        for model_id in list(self.pipelines.keys()):
            if len(self.pipelines) > total_allowed_concurrent and (keep_model is None or keep_model != model_id):
                logging.info(f'Deleting pipe for model {model_id}, as we had {len(self.pipelines)} pipes, and only {total_allowed_concurrent} are allowed.')
                del self.pipelines[model_id]
                if model_id in self.last_pipe_scheduler:
                    del self.last_pipe_scheduler[model_id]
                if model_id in self.last_pipe_type:
                    del self.last_pipe_type[model_id]
        self.clear_cuda_cache()

    def clear_cuda_cache(self):
        gc.collect()
        if config.get_cuda_cache_clear_toggle():
            logging.info("Clearing the CUDA cache...")
            torch.cuda.empty_cache()
            torch.clear_autocast_cache()
        else:
            logging.debug(
                f"NOT clearing CUDA cache. Config option `cuda_cache_clear` is not set, or is False."
            )

    def set_scheduler(self, pipe, user_config=None, scheduler_config: dict = None):
        if scheduler_config is None:
            logging.debug(f"Not setting scheduler_config parameters.")
            return
        if "name" not in scheduler_config:
            raise ValueError(f"Scheduler config must have a name: {scheduler_config}")
        if "scheduler" not in scheduler_config:
            raise ValueError(
                f"Scheduler config must have a scheduler: {scheduler_config}"
            )
        name = scheduler_config["name"]
        if name == "default":
            logging.debug(f"User selected the default scheduler. Not setting one.")
            return

        scheduler_name = scheduler_config["scheduler"]

        scheduler_module = self.SCHEDULER_MAPPINGS[scheduler_name]
        if scheduler_name == "DPMSolverMultistepScheduler":
            logging.debug(
                f"Setting algorithm_type to dpmsolver++ for {name} scheduler, {scheduler_name}."
            )
            pipe.scheduler = scheduler_module.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++"
            )
        else:
            pipe.scheduler = scheduler_module.from_config(pipe.scheduler.config)

    def get_controlnet_pipe(self):
        self.delete_pipes()
        pipeline = self.get_pipe(
            promptless_variation=True,
            user_config={},
            scheduler_config={"name": "controlnet"},
            model_id="saftle/urpm",
        )
        return pipeline

    def get_sdxl_refiner_pipe(self):
        self.delete_pipes(keep_model='ptx0/s2')
        pipeline = self.get_pipe(
            user_config={},
            scheduler_config={"name": "fast"},
            model_id="ptx0/s2",
        )
        return pipeline

    def enforce_zero_terminal_snr(self, betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    def patch_scheduler_betas(self, scheduler):
        scheduler.betas = self.enforce_zero_terminal_snr(scheduler.betas)
        return scheduler

    def to_accelerator(self, pipeline):
        return
        logging.debug(f"Moving pipeline to accelerator, begins.")
        is_on_gpu = next(pipeline.unet.parameters()).is_cuda
        if is_on_gpu:
            logging.warning(
                f"Requested to move pipeline to CPU, when it is already there."
            )
            return
        try:
            pipeline.to(self.device)
            logging.info(f"Moved pipeline to accelerator.")
        except Exception as e:
            logging.error(f"Could not move pipeline to accelerator: {e}")
            raise e

    def to_cpu(self, pipeline):
        return
        logging.debug(f"Moving pipeline to CPU, begins.")
        is_on_gpu = next(pipeline.unet.parameters()).is_cuda
        if not is_on_gpu:
            logging.warning(
                f"Requested to move pipeline to CPU, when it is already there."
            )
            return
        try:
            pipeline.to("cpu")
            torch.clear_autocast_cache()
            logging.info(f"Moved pipeline to CPU.")
        except Exception as e:
            logging.error(f"Could not move pipeline to CPU: {e}")
