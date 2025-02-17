import logging
from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.image_manipulation.pipeline_runners.overrides.flux import (
    flux_teacache_monkeypatch,
)
from discord_tron_client.classes.image_manipulation.pipeline_runners.overrides.accel import (
    optimize_pipeline,
)


config = AppConfig()


class FluxPipelineRunner(BasePipelineRunner):
    def __call__(self, **args):
        args["prompt"], prompt_parameters = self._extract_parameters(args["prompt"])

        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]
        # Use the prompt parameters to override args now
        enable_teacache = user_config.get("enable_teacache", False)
        enable_sageattn = user_config.get("enable_sageattn", True)
        if "enable_teacache" in prompt_parameters:
            enable_teacache = True
            del prompt_parameters["teacache"]
        args.update(prompt_parameters)
        logging.debug(f"Args (minus user_config) for Flux: {args}")
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
            # "negative_prompt",
        ]:
            if unwanted_arg in args:
                del args[unwanted_arg]

        # Convert specific arguments to desired types
        if "num_inference_steps" in args:
            args["num_inference_steps"] = int(float(args["num_inference_steps"]))
        if "guidance_scale" in args:
            args["guidance_scale_real"] = float(args["guidance_scale"])
            args["guidance_scale"] = float(user_config.get("flux_guidance_scale", 4.0))

        self.apply_adapters(user_config, model_prefix="flux")

        # Call the pipeline with arguments and return the images
        with flux_teacache_monkeypatch(
            self.pipeline, args.get("num_inference_steps"), disable=disable_teacache
        ):
        with optimize_pipeline(
            pipeline=self.pipeline,
            enable_teacache=enable_teacache,
            teacache_num_inference_steps=args.get("num_inference_steps"),
            teacache_rel_l1_thresh=float(
                prompt_parameters.get("teacache_distance", 0.6)
            ),
            enable_deepcache=False,
            deepcache_cache_interval=3,
            deepcache_cache_branch_id=0,
            deepcache_skip_mode="uniform",
            enable_sageattn=enable_sageattn,
        ):
            return self.pipeline(**args).images
