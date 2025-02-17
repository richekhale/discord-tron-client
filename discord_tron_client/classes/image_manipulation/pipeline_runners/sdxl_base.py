import logging, torch
from time import perf_counter
from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)
from discord_tron_client.classes.image_manipulation.pipeline_runners.overrides.accel import (
    optimize_pipeline,
)
from discord_tron_client.classes.app_config import AppConfig

config = AppConfig()


class SdxlBasePipelineRunner(BasePipelineRunner):
    def __call__(self, **args):
        args["prompt"], prompt_parameters = self._extract_parameters(args["prompt"])

        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]
        # Use the prompt parameters to override args now
        args.update(prompt_parameters)
        logging.debug(f"Args (minus user_config) for SDXL Base: {args}")
        if user_config.get("prompt_weighting", True) and config.enable_compel():
            # SDXL, when using prompt embeds, only generates 1 image per prompt.
            args["num_images_per_prompt"] = 1
            # Remove unwanted arguments for this condition
            for unwanted_arg in ["prompt", "negative_prompt"]:
                if unwanted_arg in args:
                    del args[unwanted_arg]
            if "clip_skip" in args and config.enable_compel():
                (
                    args["prompt_embeds"],
                    args["negative_embeds"],
                    args["pooled_embeds"],
                    args["negative_pooled_embeds"],
                ) = self.diffusion_manager.prompt_manager.process_long_prompt(
                    positive_prompt=args["prompt"],
                    negative_prompt=args["negative_prompt"],
                )
        else:
            # Remove unwanted arguments for this condition
            for unwanted_arg in [
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            ]:
                if unwanted_arg in args:
                    del args[unwanted_arg]

        # Convert specific arguments to desired types
        if "num_inference_steps" in args:
            args["num_inference_steps"] = int(float(args["num_inference_steps"]))
        if "guidance_scale" in args:
            args["guidance_scale"] = float(args["guidance_scale"])
        if "guidance_rescale" in args:
            args["guidance_rescale"] = float(args["guidance_rescale"])
        if "clip_skip" in args:
            args["clip_skip"] = int(args["clip_skip"])

        # Call the pipeline with arguments and return the images
        self.apply_adapters(user_config, fuse_adapters=True)

        start_time = perf_counter()
        with optimize_pipeline(
            pipeline=self.pipeline,
            enable_teacache=False,
            enable_deepcache=prompt_parameters.get("enable_deepcache", user_config.get("enable_deepcache", False)),
            deepcache_cache_interval=prompt_parameters.get("deepcache_interval", user_config.get("deepcache_interval", 3)),
            deepcache_cache_branch_id=prompt_parameters.get("deepcache_branch_id", user_config.get("deepcache_branch_id", 0)),
            deepcache_skip_mode=prompt_parameters.get("deepcache_skip_mode", user_config.get("deepcache_skip_mode", "uniform")),
        ):
            result = self.pipeline(**args).images
        torch.cuda.synchronize()
        end_time = perf_counter()
        self.generation_time = end_time - start_time
        if hasattr(self.pipeline, "deepcache_helper"):
            self.pipeline.deepcache_helper.disable()

        return result
