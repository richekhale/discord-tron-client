import logging
from typing import Any

from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)


class ZImagePipelineRunner(BasePipelineRunner):
    def __call__(self, **args: Any):
        # Extract inline parameters from the prompt and merge them into args
        args["prompt"], prompt_parameters = self._extract_parameters(args["prompt"])
        user_config = args.pop("user_config", None)
        args.update(prompt_parameters)

        # Remove unsupported arguments for the Z-Image pipeline
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

        # Normalize expected types
        if "num_inference_steps" in args:
            args["num_inference_steps"] = int(float(args["num_inference_steps"]))
        if "guidance_scale" in args:
            args["guidance_scale"] = float(args["guidance_scale"])
        if "height" in args:
            args["height"] = int(args["height"])
        if "width" in args:
            args["width"] = int(args["width"])

        # Z-Image currently only supports a single image per prompt
        args["num_images_per_prompt"] = 1
        self.apply_adapters(user_config)

        logging.debug(f"Args (minus user_config) for Z-Image: {args}")
        return self.pipeline(**args).images
