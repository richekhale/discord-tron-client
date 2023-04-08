from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline
import torch

from discord_tron_client.classes.hardware import HardwareInfo
hardware = HardwareInfo()

class DiffusionPipelineManager:
    def __init__(self):
        self.pipelines = {}
        hw_limits = hardware.get_hardware_limits()
        self.torch_dtype = torch.float16
        if hw_limits["gpu"] >= 16:
            self.torch_dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_pipe(self, pipe_id, use_attn_scaling: bool = False):
        if (use_attn_scaling):
            self.torch_dtype = torch.float16
        if pipe_id not in self.pipelines:
            self.pipelines[pipe_id] = StableDiffusionPipeline.from_pretrained(pipe_id, torch_dtype=self.torch_dtype)
        self.pipelines[pipe_id].to(self.device)
        # Disable the useless NSFW filter.
        self.pipelines[pipe_id].safety_checker = lambda images, clip_input: (images, False)
        return self.pipelines[pipe_id]