from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImageVariationPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers import DiffusionPipeline as Pipeline
from accelerate.utils import set_seed
from typing import Dict
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.app_config import AppConfig
import torch, gc, logging, diffusers

hardware = HardwareInfo()
config = AppConfig()


class DiffusionPipelineManager:
    PIPELINE_CLASSES = {
        "img2img": Pipeline,
        "text2img": Pipeline,
        "prompt_variation": StableDiffusionImg2ImgPipeline,
        "variation": StableDiffusionImageVariationPipeline,
        "upscaler": StableDiffusionUpscalePipeline,
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
    }

    def __init__(self):
        self.pipelines = {}
        hw_limits = hardware.get_hardware_limits()
        self.torch_dtype = torch.float16
        self.is_memory_constrained = False
        self.model_id = None
        self.img2img = False
        if hw_limits["gpu"] >= 16 and config.get_precision_bits() == 32:
            self.torch_dtype = torch.float32
        if hw_limits["gpu"] <= 16:
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
                gc.collect()
                torch.clear_autocast_cache()
            except Exception as e:
                logging.error(f"Error when deleting pipe: {e}")

    def create_pipeline(self, model_id: str, pipe_type: str) -> Pipeline:
        pipeline_class = self.PIPELINE_CLASSES[pipe_type]
        if pipe_type in ["img2img", "text2img"]:
            # Use the long prompt weighting pipeline.
            logging.debug(f"Creating a LPW pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                custom_pipeline="lpw_stable_diffusion",
            )
        else:
            logging.debug(f"Using standard pipeline for {model_id}")
            pipeline = pipeline_class.from_pretrained(
                model_id, torch_dtype=self.torch_dtype
            )
        if hasattr(pipeline, "safety_checker") and pipeline.safety_checker is not None:
            pipeline.safety_checker = lambda images, clip_input: (images, False)
        return pipeline

    def get_pipe(
        self,
        user_config: dict,
        scheduler_config: dict,
        resolution: dict,
        model_id: str,
        img2img: bool = False,
        prompt_variation: bool = False,
        variation: bool = False,
        upscaler: bool = False,
    ) -> Pipeline:
        self.delete_pipes(keep_model=model_id)
        logging.info("Generating a new pipe...")
        if self.is_memory_constrained:
            self.torch_dtype = torch.float16

        pipe_type = (
            "img2img"
            if img2img
            else "prompt_variation"
            if prompt_variation
            else "variation"
            if variation
            else "upscaler"
            if upscaler
            else "text2img"
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
            model_id in self.last_pipe_scheduler
            and self.last_pipe_scheduler[model_id] != scheduler_config["name"]
        ):
            logging.warn(
                f"Clearing out an incorrect pipeline and scheduler, for the same model. Going from {self.last_pipe_scheduler[model_id]} to {scheduler_config['name']}. Model: {model_id}"
            )
            self.clear_pipeline(model_id)

        if model_id not in self.pipelines:
            logging.debug(f"Creating pipeline type {pipe_type} for model {model_id}")
            self.pipelines[model_id] = self.create_pipeline(model_id, pipe_type)
            if pipe_type in ["variation"]:
                # The lambda diffusers kind of suck ass.
                self.set_scheduler(
                    pipe=self.pipelines[model_id],
                    user_config=None,
                    scheduler_config=scheduler_config,
                )
            if pipe_type in ["prompt_variation", "img2img"]:
                self.set_scheduler(
                    pipe=self.pipelines[model_id],
                    user_config=None,
                    scheduler_config=scheduler_config,
                )
                self.pipelines[model_id].enable_xformers_memory_efficient_attention(
                    True
                )
                self.pipelines[model_id].set_use_memory_efficient_attention_xformers(
                    True
                )
            if pipe_type in ["upscaler", "text2img"]:
                # Set the use of xformers library so that we can efficiently generate and upscale images.
                # @see https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/discussions/2
                self.set_scheduler(
                    pipe=self.pipelines[model_id],
                    user_config=None,
                    scheduler_config=scheduler_config,
                )
                self.pipelines[model_id].enable_xformers_memory_efficient_attention(
                    True
                )
                self.pipelines[model_id].set_use_memory_efficient_attention_xformers(
                    True
                )
        # This must happen here, or mem savings are minimal.
        self.pipelines[model_id].to(self.device)
        self.last_pipe_type[model_id] = pipe_type
        self.last_pipe_scheduler[model_id] = scheduler_config["name"]
        enable_tiling = user_config.get("enable_tiling", True)
        if enable_tiling:
            self.pipelines[model_id].vae.enable_tiling()
        else:
            self.pipelines[model_id].vae.disable_tiling()
        return self.pipelines[model_id]

    def get_variation_pipe(self, model_id):
        # Make way for the variation queen.
        logging.info("Clearing other poops.")
        self.delete_pipes(keep_model=model_id)
        logging.info("Generating a new img2img pipe...")
        self.pipelines[
            model_id
        ] = StableDiffusionImageVariationPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=self.torch_dtype,
            revision="v2.0",
        )
        from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        self.pipelines[model_id].enable_xformers_memory_efficient_attention()
        self.pipelines[model_id].safety_checker = lambda images, clip_input: (
            images,
            False,
        )
        self.pipelines[model_id].to(self.device)
        logging.info("Return the pipe...")
        return self.pipelines[model_id]

    def delete_pipes(self, keep_model: str = None):
        total_allowed_concurrent = hardware.get_concurrent_pipe_count()
        keys_to_delete = [
            pipeline
            for pipeline in self.pipelines
            if keep_model is None or pipeline != keep_model
        ]
        active_pipes = 1
        for key in keys_to_delete:
            if active_pipes >= total_allowed_concurrent:
                logging.info(
                    f"Clearing out an unwanted pipe, as we have a limit of {total_allowed_concurrent} concurrent pipes."
                )
                del self.pipelines[key]
                gc.collect()
                self.clear_cuda_cache()

    def clear_cuda_cache(self):
        if config.get_cuda_cache_clear_toggle():
            logging.info("Clearing the CUDA cache...")
            torch.cuda.empty_cache()
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
