import logging
from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)
from discord_tron_client.classes.app_config import AppConfig
from diffusers.utils.export_utils import export_to_video
from discord_tron_client.classes.image_manipulation.pipeline_runners.overrides.accel import (
    optimize_pipeline,
)

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
            "strength",
            "output_type",
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
        if "image" in args:
            # resize/crop without distorting to 768x512
            args["image"] = args["image"].resize((768, 512))

        if "decode_noise_scale" in args:
            args["decode_noise_scale"] = float(args["decode_noise_scale"])
        if "decode_timestep" in args:
            args["decode_timestep"] = float(args["decode_timestep"])

        print(f"Pipeline: {self.pipeline}")
        print(f"Pipeline args: {args}")
        enable_sageattn = user_config.get("enable_sageattn", True)
        enable_teacache = user_config.get("enable_teacache", False)
        if "enable_teacache" in prompt_parameters:
            enable_teacache = True
            del prompt_parameters["teacache"]
        with optimize_pipeline(
            pipeline=self.pipeline,
            enable_teacache=enable_teacache,
            teacache_num_inference_steps=args.get("num_inference_steps"),
            teacache_rel_l1_thresh=float(
                prompt_parameters.get(
                    "teacache_distance", user_config.get("teacache_distance", 0.6)
                )
            ),
            enable_deepcache=False,
            deepcache_cache_interval=3,
            deepcache_cache_branch_id=0,
            deepcache_skip_mode="uniform",
            enable_sageattn=enable_sageattn,
        ):
            pipeline_output = self.pipeline(**args).frames[0]
        video_path = export_to_video(pipeline_output, fps=24)
        print(f"Output: {video_path}")

        return video_path
