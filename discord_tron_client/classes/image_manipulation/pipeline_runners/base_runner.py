from PIL import Image
from diffusers import DiffusionPipeline
import torch, logging, gc, re, os
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.hardware import HardwareInfo
from huggingface_hub import hf_hub_download

config = AppConfig()
hardware_info = HardwareInfo()


class BasePipelineRunner:
    def __init__(self, **kwargs):
        self.loaded_adapters = {}
        self.pipeline = None
        self.pipeline_manager = None
        self.diffusion_manager = None
        if "pipeline" in kwargs:
            self.pipeline = kwargs["pipeline"]
        if "pipeline_manager" in kwargs:
            self.pipeline_manager = kwargs["pipeline_manager"]
        else:
            raise ValueError("Pipeline manager is required for pipeline runners.")
        if "diffusion_manager" in kwargs:
            self.diffusion_manager = kwargs["diffusion_manager"]
        else:
            raise ValueError("Pipeline manager is required for pipeline runners.")

    def run(self) -> Image:
        raise NotImplementedError

    def _cleanup_pipes(self, keep_model: str = None):
        logging.debug(
            f"Removing pipes from pipeline manager, via BasePipelineRunner._cleanup_pipes(keep_model={keep_model})"
        )
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
        return (
            hardware_info.should_offload() or hardware_info.should_sequential_offload()
        )

    def batch_size(self):
        return config.get_config_value("df_batch_size", 1)

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
            return prompt.replace("\u00A0", " ").replace("\u200B", " ")

        for idx, prompt in enumerate(prompts):
            prompt = normalize_prompt(prompt)

            parameters = {}
            if "--" in prompt:
                # Improved regular expression for parameter extraction
                param_pattern = r"--(\w+)=?([^--]*)"
                matches = re.findall(param_pattern, prompt)

                for key, value in matches:
                    # Clean up the value by removing any trailing spaces
                    parameters[key] = value.strip() if value.strip() != "" else True

                # Reconstruct the prompt without parameters
                prompt = re.sub(param_pattern, "", prompt).strip()

                prompts[idx] = prompt

            logging.debug(
                f"Prompt parameters extracted from prompt {prompt}: {parameters}"
            )

        return prompts[0] if len(prompts) == 1 else prompts, parameters

    def download_adapter(self, adapter_type: str, adapter_path: str):
        """download from huggingface hub if the adapter_type is not eg. lora"""

        adapter_filename = "pytorch_lora_weights.safetensors"
        cache_dir = config.get_huggingface_model_path()
        path_to_adapter = f"{cache_dir}/{self.clean_adapter_name(adapter_path)}"
        os.makedirs(path_to_adapter, exist_ok=True)
        hf_hub_download(
            repo_id=adapter_path, filename=adapter_filename, local_dir=path_to_adapter
        )

        return os.path.join(path_to_adapter, adapter_filename)

    def clean_adapter_name(self, adapter_path: str) -> str:
        return (
            adapter_path.replace("/", "_").replace("\\", "_").replace(":", "_")
        )

    def load_adapter(
        self,
        adapter_type: str,
        adapter_path: str,
        adapter_strength: float = 1.0,
        fuse_adapter: bool = False,
    ):
        """load the adapter from the path"""
        # remove / and other chars from the adapter name
        clean_adapter_name = self.clean_adapter_name(adapter_path)
        lycoris_wrapper = None
        if adapter_type == "lora":
            self.pipeline.load_lora_weights(
                pretrained_model_name_or_path=adapter_path,
                adapter_name=clean_adapter_name,
            )
            if fuse_adapter:
                self.pipeline.fuse_lora_weights(
                    adapter_names=[clean_adapter_name], lora_scale=adapter_strength
                )
        if adapter_type == "lycoris":
            from lycoris import create_lycoris_from_weights

            model_to_patch = getattr(
                self.pipeline, "transformer", getattr(self.pipeline, "unet", None)
            )
            path_to_adapter = self.download_adapter(adapter_type, adapter_path)
            adapter_filename = "pytorch_lora_weights.safetensors"
            lycoris_wrapper, _ = create_lycoris_from_weights(
                multiplier=adapter_strength, file=path_to_adapter, module=model_to_patch
            )
            if fuse_adapter:
                lycoris_wrapper.merge_to()
            else:
                # lycoris_wrapper.to(self.pipeline.transformer.device)
                lycoris_wrapper.apply_to()
                self.pipeline.to(self.pipeline_manager.device)
        self.loaded_adapters[clean_adapter_name] = {
            "adapter_type": adapter_type,
            "adapter_path": adapter_path,
            "adapter_strength": adapter_strength,
            "is_fused": fuse_adapter,
            "lycoris_wrapper": lycoris_wrapper,
        }

    def clear_adapters(self):
        """remove any loaded_adapters from the pipeline"""
        loaded_adapters = self.loaded_adapters.items()
        for clean_adapter_name, config in loaded_adapters:
            if config.get("adapter_type") == "lora":
                if config.get("is_fused", False):
                    self.pipeline.unfuse_lora()
                self.pipeline.unload_lora_weights(clean_adapter_name)
            if config.get("adapter_type") == "lycoris":
                lycoris_wrapper = config.get("lycoris_wrapper")
                if not lycoris_wrapper:
                    continue
                lycoris_wrapper.restore()
            self.loaded_adapters[clean_adapter_name] = None
        self.pipeline.unload_lora_weights()
        self.loaded_adapters = {}
