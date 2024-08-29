from discord_tron_client.classes.image_manipulation.pipeline_runners.base_runner import (
    BasePipelineRunner,
)
import logging


class KandinskyTwoTwoPipelineRunner(BasePipelineRunner):
    def __call__(self, **args):
        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]

        prompt = args.get("prompt", "")
        negative_prompt = args.get("negative_prompt", "")
        self.clear_cuda_cache()
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=args.get("height", 768),
            width=args.get("width", 768),
        ).images
