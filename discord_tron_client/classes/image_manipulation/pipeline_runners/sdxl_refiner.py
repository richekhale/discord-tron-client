from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)


class SdxlRefinerPipelineRunner(BasePipelineRunner):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, **args):
        # Set all defaults at once
        user_config = args.get("user_config", None)
        default_values = {
            "prompt": None,
            "negative_prompt": None,
            "num_inference_steps": None,
            "guidance_scale": None,
            "guidance_rescale": None,
            "num_images_per_prompt": None,
            "output_type": None,
            "aesthetic_score": None,
            "negative_aesthetic_score": None,
            "generator": None,
            "prompt_embeds": None,
            "negative_prompt_embeds": None,
            "pooled_prompt_embeds": None,
            "negative_pooled_prompt_embeds": None,
            "strength": None,
            "denoising_start": None,
            "denoising_end": None,
        }
        runtime_args = {
            **default_values,
            **args,
        }  # merge the two dictionaries, with priority on args
        if args["image"] is not None:
            runtime_args["image"] = args["image"]
        if (
            "width" in args
            and args["width"] is not None
            and "height" in args
            and args["height"] is not None
        ):
            runtime_args["width"] = args["width"]
            runtime_args["height"] = args["height"]

        # Currently, it seems like the refiner's prompt weighting is broken.
        # We are disabling it by default.
        if user_config is not None and user_config.get(
            "refiner_prompt_weighting", False
        ):
            runtime_args[
                "num_images_per_prompt"
            ] = 1  # SDXL, when using prompt embeds, only generates 1 image per prompt.
            return self.pipeline(**runtime_args).images
        else:
            for unwanted_arg in [
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            ]:
                if unwanted_arg in runtime_args:
                    del runtime_args[unwanted_arg]
            return self.pipeline(**runtime_args).images
