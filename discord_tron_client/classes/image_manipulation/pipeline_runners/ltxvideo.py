import logging
from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)
from discord_tron_client.classes.app_config import AppConfig
from diffusers.utils.export_utils import export_to_video
config = AppConfig()


class LtxVideoPipelineRunner(BasePipelineRunner):
    def __call__(self, **args):
        
        args["prompt"], prompt_parameters = self._extract_parameters(args["prompt"])

        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]
        args.update(prompt_parameters)
        logging.debug(f"Args (minus user_config) for Sana: {args}")
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
            "num_images_per_prompt",
        ]:
            if unwanted_arg in args:
                del args[unwanted_arg]

        # Convert specific arguments to desired types
        if "num_inference_steps" in args:
            args["num_inference_steps"] = int(float(args["num_inference_steps"]))
        if "guidance_scale" in args:
            args["guidance_scale"] = float(args["guidance_scale"])
        
        args["width"] = 768
        args["height"] = 512

        print(f"Pipeline: {self.pipeline}")
        print(f"Pipeline args: {args}")
        pipeline_output = self.pipeline(**args).frames[0]
        video_path = export_to_video(pipeline_output, fps=24)
        print(f"Output: {video_path}")

        return video_path
