from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)


class SdxlRefinerPipelineRunner(BasePipelineRunner):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, **args):
        # Set all defaults at once
        user_config = args.get("user_config", None)
        del args["user_config"] # This doesn't get passed to Diffusers.

        # Currently, it seems like the refiner's prompt weighting is broken.
        # We are disabling it by default.
        if user_config is not None and user_config.get(
            "refiner_prompt_weighting", False
        ):
            args[
                "num_images_per_prompt"
            ] = 1  # SDXL, when using prompt embeds, only generates 1 image per prompt.
            return self.pipeline(**args).images
        else:
            for unwanted_arg in [
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            ]:
                if unwanted_arg in args:
                    del args[unwanted_arg]
            return self.pipeline(**args).images
