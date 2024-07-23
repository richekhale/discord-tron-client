import logging, random, torch
from discord_tron_client.classes.image_manipulation.pipeline_runners import BasePipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.overrides.pixart import PixArtSigmaPipeline
from diffusers.models import PixArtTransformer2DModel
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()
class PixArtPipelineRunner(BasePipelineRunner):
    refiner_pipeline = None
    def __call__(self, **args):
        args["prompt"], prompt_parameters = self._extract_parameters(args["prompt"])

        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        output_type = "pil"
        should_run_stage_2 = False
        split_schedule_interval = None
        if user_config.get("latent_refiner", False) or "stage1" in user_config.get("model", "") or "vpred-zsnr" in user_config.get("model", ""):
            # If latent refiner is enabled, we need to run the refiner pipeline first
            output_type = "latent"
            should_run_stage_2 = True
            split_schedule_interval = 0.6
            if self.refiner_pipeline is None:
                self.get_refiner_pipeline()
        del args["user_config"]
        # Use the prompt parameters to override args now
        args.update(prompt_parameters)
        logging.debug(f'Args (minus user_config) for PixArt: {args}')
        # Remove unwanted arguments for this condition
        for unwanted_arg in ["prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds", "clip_skip", "denoising_start", "denoising_end"]:
            if unwanted_arg in args:
                del args[unwanted_arg]

        # Convert specific arguments to desired types
        if "num_inference_steps" in args:
            args["num_inference_steps"] = int(float(args["num_inference_steps"]))
        if "guidance_scale" in args:
            args["guidance_scale"] = float(args["guidance_scale"])
        if "output_type" in args:
            del args["output_type"]
        if "denoising_end" in args:
            del args["denoising_end"]
        if "denoising_start" in args:
            del args["denoising_start"]
        stage_2_guidance = user_config.get("refiner_guidance", args["guidance_scale"])
        if "stage_2_guidance" in prompt_parameters:
            stage_2_guidance = prompt_parameters["stage_2_guidance"]
            del prompt_parameters["stage_2_guidance"]

        args["prompt"] = args["prompt"].strip()
        user_seed = user_config.get("seed", 0)
        if user_seed == -1:
            user_seed = random.randint(0, 1000000)
        elif user_seed == 0:
            import time
            user_seed = time.time()
        args["generator"] = self.diffusion_manager._get_generator(user_config=user_config)
        
        # Call the pipeline with arguments and return the images
        args = {"output_type": output_type, "denoising_end": split_schedule_interval, **args}
        logging.debug(f'Running base pipeline with final adjusted args: {args}')
        base_images = self.pipeline(**args).images
        if should_run_stage_2:
            args["image"] = None
            args["denoising_end"] = None
            args["denoising_start"] = split_schedule_interval
            args["output_type"] = "pil"
            args["guidance_scale"] = float(stage_2_guidance)
            args["strength"] = None
            logging.debug(f'Running refiner pipeline with adjusted args: {args}')
            refiner_images = self.refiner_pipeline(latents=base_images, **args).images
            return refiner_images
        return base_images

    def get_refiner_pipeline(self):
        if self.refiner_pipeline is None:
            device = self.pipeline.transformer.device
            dtype = self.pipeline.transformer.dtype
            refiner_model = config.get_config_value("refiner_model", "ptx0/pixart-900m-1024-ft-v0.7-stage2")
            self.refiner_pipeline = PixArtSigmaPipeline.from_pretrained(
                pretrained_model_name_or_path=refiner_model,
                torch_dtype=dtype,
                **self.pipeline.components
            )
            self.refiner_pipeline.transformer = PixArtTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path=refiner_model,
                torch_dtype=dtype,
                subfolder="transformer"
            ).to(device)
        return self.refiner_pipeline