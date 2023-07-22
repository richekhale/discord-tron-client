from PIL import Image
from diffusers import DiffusionPipeline
import torch, logging, gc
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.hardware import HardwareInfo
config = AppConfig()
hardware_info = HardwareInfo()

class BasePipelineRunner:
    def __init__(self, **kwargs):
        self.pipeline = None
        self.pipeline_manager = None
        self.diffusion_manager = None
        if 'pipeline' in kwargs:
            self.pipeline = kwargs['pipeline']
        if 'pipeline_manager' in kwargs:
            self.pipeline_manager = kwargs['pipeline_manager']
        else:
            raise ValueError('Pipeline manager is required for pipeline runners.')
        if 'diffusion_manager' in kwargs:
            self.diffusion_manager = kwargs['diffusion_manager']
        else:
            raise ValueError('Pipeline manager is required for pipeline runners.')


    def run(self) -> Image:
        raise NotImplementedError
    
    def _cleanup_pipes(self, keep_model: str = None):
        logging.debug(f'Removing pipes from pipeline manager, via BasePipelineRunner._cleanup_pipes(keep_model={keep_model})')
        return self.pipeline_manager.delete_pipes(keep_model=keep_model)

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
    def should_offload(self):
        return hardware_info.should_offload() or hardware_info.should_sequential_offload()