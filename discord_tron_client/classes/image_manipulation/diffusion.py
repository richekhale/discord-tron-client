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
    AutoPipelineForText2Image,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import DiffusionPipeline as Pipeline
from typing import Dict
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.image_manipulation.face_upscale import (
    get_upscaler,
    use_upscaler,
)
from PIL import Image
import torch, gc, logging, diffusers, transformers

torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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
        "kandinsky-2.2": AutoPipelineForText2Image,
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

    def create_pipeline(self, model_id: str, pipe_type: str, use_safetensors: bool = True, custom_text_encoder = None, safety_modules: dict = None) -> Pipeline:
        pipeline_class = self.PIPELINE_CLASSES[pipe_type]
        extra_args = {
            'feature_extractor': None,
            'safety_checker': None,
            'requires_safety_checker': None,
        }
        if custom_text_encoder is not None and custom_text_encoder == -1:
            # Disable text encoder.
            extra_args["text_encoder"] = None
        elif custom_text_encoder is not None:
            # Use a custom text encoder.
            extra_args["text_encoder"] = custom_text_encoder
        if safety_modules is not None:
            for key in safety_modules:
                extra_args[key] = safety_modules[key]
        if pipe_type in ["variation", "upscaler"]:
            # Variation uses ControlNet stuff.
            logging.debug(f"Creating a ControlNet model for {model_id}")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=self.torch_dtype
            )
            logging.debug(
                f"Passing the ControlNet into a StableDiffusionControlNetPipeline for {model_id}"
            )
            logging.debug(
                f"Passing args into ControlNet: {extra_args} for {model_id}"
            )
            pipeline = self.PIPELINE_CLASSES["text2img"].from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                custom_pipeline="stable_diffusion_controlnet_img2img",
                controlnet=controlnet,
                use_safetensors=use_safetensors,
                **extra_args
            )
        elif pipe_type in ["prompt_variation"]:
            # Use the long prompt weighting pipeline.
            logging.debug(f"Creating a LPW pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=use_safetensors,
                **extra_args
            )
        elif pipe_type in ["text2img"]:
            logging.debug(f"Creating a txt2img pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=use_safetensors,
                use_auth_token=config.get_huggingface_api_key(),
                variant=config.get_config_value('model_default_variant', None),
                **extra_args
            )
            logging.debug(f"Model config: {pipeline.config}")
        else:
            logging.debug(f"Using standard pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id, torch_dtype=self.torch_dtype,
                use_safetensors=use_safetensors,
                use_auth_token=config.get_huggingface_api_key(),
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
        custom_text_encoder = None,
        safety_modules: dict = None,
        use_safetensors: bool = True
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
            logging.debug(f"Creating pipeline type {pipe_type} for model {model_id} with custom_text_encoder {type(custom_text_encoder)}")
            self.pipelines[model_id] = self.create_pipeline(model_id, pipe_type, use_safetensors=use_safetensors, custom_text_encoder=custom_text_encoder, safety_modules=safety_modules)
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
            if hasattr(self.pipelines[model_id], 'unet'):
                self.pipelines[model_id].unet.to(memory_format=torch.channels_last)
                self.pipelines[model_id].unet.set_attn_processor(AttnProcessor2_0()) # https://huggingface.co/docs/diffusers/optimization/torch2.0
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
            else:
                logging.info(
                    f"Moving pipe to CUDA early, because no offloading is being used."
                )
                self.pipelines[model_id].to(self.device)
                if config.enable_compile() and hasattr(self.pipelines[model_id], 'unet'):
                    torch._dynamo.config.suppress_errors = True
                    self.pipelines[model_id].unet = torch.compile(
                        self.pipelines[model_id].unet,
                        mode="reduce-overhead",
                        fullgraph=True,
                    )
                if hasattr(self.pipelines[model_id], 'controlnet') and config.enable_compile():
                    self.pipelines[model_id].controlnet = torch.compile(self.pipelines[model_id].controlnet, fullgraph=True)
                if hasattr(self.pipelines[model_id], 'text_encoder') and type(self.pipelines[model_id].text_encoder) == transformers.T5EncoderModel and config.enable_compile():
                    logging.info('Found T5 encoder model. Compiling...')
                    self.pipelines[model_id].text_encoder = torch.compile(
                        self.pipelines[model_id].text_encoder,
                        fullgraph=True,
                    )
                elif hasattr(self.pipelines[model_id], 'text_encoder'):
                    logging.warning(f'Torch compile on text encoder type {type(self.pipelines[model_id].text_encoder)} is not yet supported.')
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
            model_id="emilianJR/epiCRealism",
            use_safetensors=False
        )
        return pipeline

    def get_sdxl_refiner_pipe(self):
        refiner_model = config.get_config_value('refiner_model', 'stabilityai/stable-diffusion-xl-refiner-1.0')
        self.delete_pipes(keep_model=refiner_model)
        pipeline = self.get_pipe(
            user_config={},
            scheduler_config={"name": "fast"},
            model_id=refiner_model,
        )
        pipeline.vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16, use_safetensors=True, use_auth_token=config.get_huggingface_api_key()).to(self.device)
        return pipeline