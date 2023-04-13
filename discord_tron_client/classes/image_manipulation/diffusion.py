from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionSAGPipeline
from accelerate.utils import set_seed
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.app_config import AppConfig
import torch, gc, logging

hardware = HardwareInfo()
config = AppConfig()

class DiffusionPipelineManager:
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

    def get_pipe(self, model_id, img2img: bool = False, SAG: bool = False):
        gc.collect()
        logging.info("Generating a new text2img pipe...")
        if (self.use_attn_scaling):
            self.torch_dtype = torch.float16
        # Clear every model from memory except the ones we're working with most recently, as many as the GPU will hold.
        self.delete_pipes(keep_model=model_id)
        if img2img:
            # Basic StableDiffusionImg2Img pipeline flow. Not great results.
            if self.last_pipe_type[model_id] != "img2img" and model_id in self.pipelines:
                self.pipelines[model_id].clear()
            if model_id not in self.pipelines:
                self.pipelines[model_id] = StableDiffusionImg2ImgPipeline.from_pretrained(
                    pretrained_model_name_or_path=model_id, torch_dtype=self.torch_dtype
                )
        elif SAG:
            # Self-assisted guidance pipeline flow.
            if self.last_pipe_type[model_id] != "SAG" and model_id in self.pipelines:
                logging.warn(f"Clearing out an incorrect pipeline type for the same model. Going from {self.last_pipe_type[model_id]} to SAG. Model: {model_id}")
                self.pipelines[model_id].clear()
            if model_id not in self.pipelines:
                self.pipelines[model_id] = StableDiffusionSAGPipeline.from_pretrained(
                    pretrained_model_name_or_path=model_id, torch_dtype=self.torch_dtype
                )
        else:
            # Use a vanilla StableDiffusion Pipeline flow.
            if self.last_pipe_type[model_id] != "txt2img" and model_id in self.pipelines:
                logging.warn(f"Clearing out an incorrect pipeline type for the same model. Going from {self.last_pipe_type[model_id]} to SAG. Model: {model_id}")
                self.pipelines[model_id].clear()
            if model_id not in self.pipelines:
                self.pipelines[model_id] = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.torch_dtype)
        self.pipelines[model_id].to(self.device)
        # Disable the useless NSFW filter.
        self.pipelines[model_id].safety_checker = lambda images, clip_input: (images, False)
        # Set self.last_pipe_type by the values of img2img and SAG. A negative value of both, means last_pipe_type is "text2img"
        if img2img:
            self.last_pipe_type[model_id] = "img2img"
        elif SAG:
            self.last_pipe_type[model_id] = "SAG"
        else:
            self.last_pipe_type[model_id] = "text2img"
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
        