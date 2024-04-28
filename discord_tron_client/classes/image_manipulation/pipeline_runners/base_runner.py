from PIL import Image
from diffusers import DiffusionPipeline
import torch, logging, gc, re
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
    
    def batch_size(self):
        return config.get_config_value('df_batch_size', 1)

    def _extract_parameters(self, prompts: str) -> tuple:
        """
        Extracts key-value parameters from a prompt string using a more robust regular expression.

        Args:
            prompt (str): The prompt string potentially containing parameters.

        Returns:
            tuple: A tuple containing: 
                - The original prompt with parameters removed.
                - A dictionary of extracted key-value parameters.
        """
        if type(prompts) is not list:
            prompts = [prompts]

        def normalize_prompt(prompt):
            return prompt.replace('\u00A0', ' ').replace('\u200B', ' ')

        for idx, prompt in enumerate(prompts):
            prompt = normalize_prompt(prompt)

            parameters = {}
            if "--" in prompt:
                # Improved regular expression for parameter extraction
                param_pattern = r"--(\w+)=?([^--]*)"
                matches = re.findall(param_pattern, prompt)

                for key, value in matches:
                    # Clean up the value by removing any trailing spaces
                    parameters[key] = value.strip()

                # Reconstruct the prompt without parameters
                prompt = re.sub(param_pattern, '', prompt).strip()

                prompts[idx] = prompt

            logging.debug(f"Prompt parameters extracted from prompt {prompt}: {parameters}")

        return prompts[0] if len(prompts) == 1 else prompts, parameters
