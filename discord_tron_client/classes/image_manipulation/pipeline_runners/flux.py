import logging
from discord_tron_client.classes.image_manipulation.pipeline_runners import BasePipelineRunner
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()

class FluxPipelineRunner(BasePipelineRunner):
    def __call__(self, **args):
        args["prompt"], prompt_parameters = self._extract_parameters(args["prompt"])

        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]
        # Use the prompt parameters to override args now
        args.update(prompt_parameters)
        logging.debug(f'Args (minus user_config) for SD3: {args}')
        # Remove unwanted arguments for this condition
        for unwanted_arg in ["prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds", "guidance_rescale", "clip_skip", "denoising_start", "denoising_end", "negative_prompt"]:
            if unwanted_arg in args:
                del args[unwanted_arg]

        # Convert specific arguments to desired types
        if "num_inference_steps" in args:
            args["num_inference_steps"] = int(float(args["num_inference_steps"]))
        if "guidance_scale" in args:
            args["guidance_scale"] = float(args["guidance_scale"])
        
        # we will apply user LoRAs one at a time. the lora name can be split optionally with a : at the end so that <lora_path>:<strength> are set.
        range = range(1, 11)
        self.clear_adapters()
        for i in range:
            user_adapter = user_config.get(f"flux_adapter_{i}", None)
            if user_adapter is not None:
                pieces = user_adapter.split(":")
                adapter_strength = 1
                adapter_type = "lora"
                if len(pieces) == 1:
                    adapter_path = pieces[0]
                elif len(pieces) == 2:
                    adapter_type, adapter_path = pieces
                elif len(pieces) == 3:
                    adapter_type, adapter_path, adapter_strength = pieces
                try:
                    self.load_adapter(adapter_type, adapter_path, adapter_strength, fuse_adapter=False)
                except:
                    logging.error(f"Failed to download adapter {adapter_path}")
                    continue
                
        
        # Call the pipeline with arguments and return the images
        return self.pipeline(**args).images
