import logging
from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)
from discord_tron_client.classes.app_config import AppConfig

config = AppConfig()


class SD3PipelineRunner(BasePipelineRunner):
    def __call__(self, **args):
        args["prompt"], prompt_parameters = self._extract_parameters(args["prompt"])

        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]
        args["skip_guidance_layers"] = [7, 8, 9]
        args["max_sequence_length"] = 154
        # Use the prompt parameters to override args now
        args.update(prompt_parameters)
        logging.debug(f"Args (minus user_config) for SD3: {args}")
        # Remove unwanted arguments for this condition
        for unwanted_arg in [
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "guidance_rescale",
            "clip_skip",
            "denoising_start",
            "denoising_end",
        ]:
            if unwanted_arg in args:
                del args[unwanted_arg]

        # Convert specific arguments to desired types
        if "num_inference_steps" in args:
            args["num_inference_steps"] = int(float(args["num_inference_steps"]))
        if "guidance_scale" in args:
            args["guidance_scale"] = float(args["guidance_scale"])
        if "max_sequence_length" in args:
            args["max_sequence_length"] = int(float(args["max_sequence_length"]))
        if "skip_guidance_start" in args:
            args["skip_layer_guidance_start"] = float(args["skip_layer_guidance_start"])
        if "skip_layer_guidance_end" in args:
            args["skip_layer_guidance_end"] = float(args["skip_layer_guidance_end"])
        if "skip_layer_guidance_scale" in args:
            args["skip_layer_guidance_end"] = float(args["skip_layer_guidance_scale"])


        self.apply_adapters(user_config)

        # Call the pipeline with arguments and return the images
        return self.pipeline(**args).images
