from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionSAGPipeline
from diffusers import DiffusionPipeline as Pipeline
from accelerate.utils import set_seed
from typing import Dict
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.app_config import AppConfig
import torch, gc, logging

hardware = HardwareInfo()
config = AppConfig()

class DiffusionPipelineManager:
    PIPELINE_CLASSES = {
        "img2img": StableDiffusionImg2ImgPipeline,
        "SAG": StableDiffusionSAGPipeline,
        "text2img": StableDiffusionPipeline,
        "prompt_variation": StableDiffusionImg2ImgPipeline,
        "variation": StableDiffusionImageVariationPipeline,
    }
    def __init__(self):
        self.pipelines = {}
        hw_limits = hardware.get_hardware_limits()
        self.torch_dtype = torch.float16
        self.variation_attn_scaling = False
        self.use_attn_scaling = False
        self.model_id = None
        self.img2img = False
        self.SAG = False
        if hw_limits["gpu"] >= 16 and config.get_precision_bits() == 32:
            self.torch_dtype = torch.float32
        if hw_limits["gpu"] <= 10:
            self.variation_attn_scaling = True
            self.use_attn_scaling = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_pipe_type = {}        # { "model_id": "txt2img", ... }
        self.pipelines: Dict[str, Pipeline] = {}
        self.last_pipe_type: Dict[str, str] = {}

    def clear_pipeline(self, model_id: str) -> None:
        if model_id in self.pipelines:
            self.pipelines[model_id].clear()

    def create_pipeline(self, model_id: str, pipe_type: str) -> Pipeline:
        pipeline_class = self.PIPELINE_CLASSES[pipe_type]
        pipeline = pipeline_class.from_pretrained(model_id, torch_dtype=self.torch_dtype)
        pipeline.to(self.device)
        if pipeline.safety_checker is not None:
            pipeline.safety_checker = lambda images, clip_input: (images, False)
        return pipeline

    def get_pipe(self, model_id: str, img2img: bool = False, SAG: bool = False, prompt_variation: bool = False, variation: bool = False) -> Pipeline:
        gc.collect()
        logging.info("Generating a new text2img pipe...")

        if self.use_attn_scaling:
            self.torch_dtype = torch.float16

        pipe_type = "img2img" if img2img else "SAG" if SAG else "prompt_variation" if prompt_variation else "variation" if variation else "text2img"
        
        if model_id in self.last_pipe_type and self.last_pipe_type[model_id] != pipe_type:
            logging.warn(f"Clearing out an incorrect pipeline type for the same model. Going from {self.last_pipe_type[model_id]} to {pipe_type}. Model: {model_id}")
            self.clear_pipeline(model_id)

        if model_id not in self.pipelines:
            self.pipelines[model_id] = self.create_pipeline(model_id, pipe_type)
            if pipe_type in ["prompt_variation", "variation"]:
                if self.variation_attn_scaling:
                    logging.info("Using attention scaling, due to hardware limits! This will make generation run more slowly, but it will be less likely to run out of memory.")
                    self.pipelines[model_id].enable_sequential_cpu_offload()
                    self.pipelines[model_id].enable_attention_slicing(1)

        self.last_pipe_type[model_id] = pipe_type

        return self.pipelines[model_id]

    def get_prompt_variation_pipe(self, model_id):
        # Make way for the variation queen.
        logging.info("Clearing other poops.")
        self.delete_pipes(keep_model=model_id)
        logging.info("Generating a new text2img pipe...")
        
        self.pipelines[model_id] = StableDiffusionImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id, torch_dtype=self.torch_dtype
        )
        if (self.variation_attn_scaling):
            logging.info("Using attention scaling, due to hardware limits! This will make generation run more slowly, but it will be less likely to run out of memory.")
            self.pipelines[model_id].enable_sequential_cpu_offload()
            self.pipelines[model_id].enable_attention_slicing(1)
        self.pipelines[model_id].safety_checker = lambda images, clip_input: (images, False)
        logging.info("Return the pipe...")
        return self.pipelines[model_id]

    def get_variation_pipe(self, model_id):
        # Make way for the variation queen.
        logging.info("Clearing other poops.")
        self.delete_pipes(keep_model=model_id)
        logging.info("Generating a new img2img pipe...")
        self.pipelines[model_id] = StableDiffusionImageVariationPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id, torch_dtype=self.torch_dtype
        )
        if (self.variation_attn_scaling):
            logging.info("Using attention scaling, due to hardware limits! This will make generation run more slowly, but it will be less likely to run out of memory.")
            self.pipelines[model_id].enable_sequential_cpu_offload()
            self.pipelines[model_id].enable_attention_slicing(1)
        self.pipelines[model_id].safety_checker = lambda images, clip_input: (images, False)
        logging.info("Return the pipe...")
        return self.pipelines[model_id]
    
    def delete_pipes(self, keep_model: str = None):
        for pipeline in self.pipelines:
            if keep_model is None or pipeline != keep_model:
                del self.pipelines[pipeline]
                gc.collect()

        if config.get_cuda_cache_clear_toggle():
            logging.info("Clearing the CUDA cache...")
            torch.cuda.empty_cache()
        