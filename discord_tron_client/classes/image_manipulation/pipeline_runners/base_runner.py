from PIL import Image
from diffusers import DiffusionPipeline
import torch, logging, gc
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()
class BasePipelineRunner:
    def __init__(self, pipeline: DiffusionPipeline):
        self.pipeline = pipeline

    def run(self) -> Image:
        raise NotImplementedError
    
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