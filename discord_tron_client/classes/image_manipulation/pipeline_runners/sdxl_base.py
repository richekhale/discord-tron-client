import logging
from discord_tron_client.classes.image_manipulation.pipeline_runners import BasePipelineRunner
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()
class SdxlBasePipelineRunner(BasePipelineRunner):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, **args):
        args["prompt"], prompt_parameters = self._extract_parameters(args["prompt"])
        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]
        logging.debug(f'Args (minus user_config) for SDXL Base: {args}')
        if user_config.get("prompt_weighting", True) and config.enable_compel():
            # SDXL, when using prompt embeds, only generates 1 image per prompt.
            args["num_images_per_prompt"] = 1
            # Remove unwanted arguments for this condition
            for unwanted_arg in ["prompt", "negative_prompt"]:
                if unwanted_arg in args:
                    del args[unwanted_arg]
        else:
            # Remove unwanted arguments for this condition
            for unwanted_arg in ["prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"]:
                if unwanted_arg in args:
                    del args[unwanted_arg]
        
        # Convert specific arguments to desired types
        if "num_inference_steps" in args:
            args["num_inference_steps"] = int(float(args["num_inference_steps"]))
        if "guidance_scale" in args:
            args["guidance_scale"] = float(args["guidance_scale"])
        if "guidance_rescale" in args:
            args["guidance_rescale"] = float(args["guidance_rescale"])

        # Use the prompt parameters to override args now
        args.update(prompt_parameters)
        
        # Call the pipeline with arguments and return the images
        return self.pipeline(**args).images
