from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline, StableDiffusionImg2ImgPipeline
from discord_tron_client.classes.hardware import HardwareInfo
import torch, gc, logging

hardware = HardwareInfo()

class DiffusionPipelineManager:
    def __init__(self):
        self.pipelines = {}
        hw_limits = hardware.get_hardware_limits()
        self.torch_dtype = torch.float16
        self.variation_attn_scaling = False
        self.use_attn_scaling = False
        if hw_limits["gpu"] >= 16:
            self.torch_dtype = torch.float32
        if hw_limits["gpu"] <= 10:
            self.variation_attn_scaling = True
            self.use_attn_scaling = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_pipe(self, pipe_id, img2img: bool = False):
        gc.collect()
        logging.info("Generating a new text2img pipe...")
        if (self.use_attn_scaling):
            self.torch_dtype = torch.float16
        if pipe_id not in self.pipelines:
            self.delete_pipes()
            if img2img:
                self.pipelines[pipe_id] = StableDiffusionImg2ImgPipeline.from_pretrained(
                    pretrained_model_name_or_path=pipe_id, torch_dtype=self.torch_dtype
                )
            else:
                self.pipelines[pipe_id] = StableDiffusionPipeline.from_pretrained(pipe_id, torch_dtype=self.torch_dtype)
        self.pipelines[pipe_id].to(self.device)
        # Disable the useless NSFW filter.
        self.pipelines[pipe_id].safety_checker = lambda images, clip_input: (images, False)
        return self.pipelines[pipe_id]

    def get_prompt_variation_pipe(self, model_id):
        # Make way for the variation queen.
        logging.info("Clearing other poops.")
        self.delete_pipes()
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
        self.delete_pipes()
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
    
    def delete_pipes(self):
        del self.pipelines
        self.pipelines = {}
        gc.collect()
        logging.info("Clearing the CUDA cache...")
        torch.cuda.empty_cache()
        