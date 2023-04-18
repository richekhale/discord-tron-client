import logging
import sys
import torch
import traceback
import time
import asyncio
from tqdm import tqdm
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.image_manipulation.resolution import ResolutionManager
from discord_tron_client.classes.image_manipulation.prompt_manipulation import PromptManipulation
from discord_tron_client.classes.tqdm_capture import TqdmCapture
from discord_tron_client.classes.discord_progress_bar import DiscordProgressBar
from discord_tron_client.message.discord import DiscordMessage
from PIL import Image

class PipelineRunner:
    def __init__(
        self,
        model_manager,
        pipeline_manager,
        app_config: AppConfig,
        user_config: dict,
        discord_msg,
        websocket,
        model_config: dict = {"sag_capable": False}
    ):
        # General AppConfig() object access.
        self.config = app_config
        self.seed = None
        main_loop = asyncio.get_event_loop()
        if main_loop is None:
            raise Exception("AppConfig.main_loop is not set!")
        # The received user_config item from TRON master.
        self.user_config = user_config
        # Managers.
        self.model_manager = model_manager
        self.pipeline_manager = pipeline_manager
        # A message template for the WebSocket events.
        self.progress_bar_message = DiscordMessage(
            websocket=websocket,
            context=discord_msg,
            module_command="edit"
        )
        # An object to manage a progress bar for Discord.
        self.progress_bar = DiscordProgressBar(
            websocket=websocket,
            websocket_message=self.progress_bar_message,
            progress_bar_steps=100,
            progress_bar_length=20,
            discord_first_message=discord_msg
        )
        self.tqdm_capture = TqdmCapture(self.progress_bar, main_loop)
        self.websocket = websocket
        self.model_config = model_config

    async def _prepare_pipe_async(
        self,
        model_id: int,
        img2img: bool = False,
        promptless_variation: bool = False,
        SAG: bool = False,
        upscaler: bool = False
    ):
        loop = asyncio.get_event_loop()
        loop_return = await loop.run_in_executor(
            AppConfig.get_image_worker_thread(),  # Use a dedicated image processing thread worker.
            self._prepare_pipe,
            model_id,
            img2img,
            promptless_variation,
            SAG,
            upscaler
        )
        return loop_return

    def _prepare_pipe(
        self,
        model_id: int,
        img2img: bool = False,
        promptless_variation: bool = False,
        SAG: bool = False,
        upscaler: bool = False
    ):
        logging.info(f"Retrieving pipe for model {model_id}")
        if not promptless_variation:
            pipe = self.pipeline_manager.get_pipe(model_id, img2img, SAG, promptless_variation, variation=False, upscaler=upscaler)
        else:
            pipe = self.pipeline_manager.get_variation_pipe(model_id)
        logging.info("Copied pipe to the local context")
        return pipe

    async def _generate_image_with_pipe_async(
        self,
        pipe,
        prompt,
        side_x: int,
        side_y: int,
        steps: int,
        negative_prompt,
        user_config: dict,
        image: Image = None,
        promptless_variation: bool = False,
        upscaler: bool = False,
        SAG: bool = False
    ):
        loop = asyncio.get_event_loop()
        loop_return = await loop.run_in_executor(
            AppConfig.get_image_worker_thread(),  # Use a dedicated image processing thread worker.
            self._generate_image_with_pipe,
            pipe,
            prompt,
            side_x,
            side_y,
            steps,
            negative_prompt,
            user_config,
            image,
            promptless_variation,
            upscaler
        )
        return loop_return

    def _generate_image_with_pipe(
        self,
        pipe,
        prompt,
        side_x: int,
        side_y: int,
        steps: int,
        negative_prompt: str,
        user_config: dict,
        image: Image = None,
        promptless_variation: bool = False,
        upscaler: bool = False
    ):
        try:
            guidance_scale = user_config.get("guidance_scale", 7.5)
            guidance_scale = min(guidance_scale, 20)

            SAG = user_config.get("enable_sag", True)
            sag_scale = user_config.get("sag_scale", 0.75)
            sag_scale = min(sag_scale, 20)
            self.gpu_power_consumption = 0.0
            generator = self._get_generator(user_config=user_config)

            prompt_embed, negative_embed = self.prompt_manager.process_long_prompt(positive_prompt=prompt, negative_prompt=negative_prompt)

            with torch.no_grad():
                with tqdm(total=steps, ncols=100, file=self.tqdm_capture) as pbar:
                    new_image = self._run_pipeline(
                        pipe,
                        prompt_embed,
                        side_x,
                        side_y,
                        steps,
                        negative_embed,
                        guidance_scale,
                        generator,
                        SAG,
                        sag_scale,
                        user_config,
                        image,
                        promptless_variation,
                        upscaler,
                        positive_prompt=prompt,
                        negative_prompt=negative_prompt
                    )
            self.gpu_power_consumption = self.tqdm_capture.gpu_power_consumption
            return new_image
        except Exception as e:
            logging.error(f"Error while generating image: {e}\n{traceback.format_exc()}")

    def _run_pipeline(
        self,
        pipe,
        prompt_embed,
        side_x: int,
        side_y: int,
        steps: int,
        negative_embed: str,
        guidance_scale: float,
        generator,
        SAG: bool,
        sag_scale: float,
        user_config: dict,
        image: Image = None,
        promptless_variation: bool = False,
        upscaler: bool = False,
        positive_prompt = "",
        negative_prompt = ""
    ):
        original_stderr = sys.stderr
        sys.stderr = self.tqdm_capture
        try:
            alt_weight_algorithm = user_config.get("alt_weight_algorithm", False)
            if not promptless_variation and image is None and not SAG:
                if not alt_weight_algorithm:
                    # Default "long prompt weighting" pipeline
                    new_image = pipe(
                        prompt=positive_prompt,
                        height=side_y,
                        width=side_x,
                        num_inference_steps=int(float(steps)),
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                else:
                    # Use the Compel library's prompt weights as input instead.
                    new_image = pipe(
                        positive_embeds=prompt_embed,
                        height=side_y,
                        width=side_x,
                        num_inference_steps=int(float(steps)),
                        negative_embeds=negative_embed,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
            elif SAG:
                new_image = pipe(
                    prompt_embeds=prompt_embed,
                    height=side_y,
                    width=side_x,
                    num_inference_steps=int(float(steps)),
                    negative_prompt_embeds=negative_embed,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    sag_scale=sag_scale,
                ).images[0]
            elif not upscaler and image is not None:
                if not alt_weight_algorithm:
                    new_image = pipe.img2img(
                        prompt=positive_prompt,
                        image=image,
                        strength=user_config["strength"],
                        num_inference_steps=int(float(steps)),
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
                else:
                    new_image = pipe(
                        prompt_embeds=prompt_embed,
                        image=image,
                        strength=user_config["strength"],
                        num_inference_steps=int(float(steps)),
                        negative_prompt_embeds=negative_embed,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
            elif promptless_variation:
                new_image = pipe(
                    height=side_y,
                    width=side_x,
                    num_inference_steps=int(float(steps)),
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
            elif upscaler:
                new_image = pipe(prompt_embeds=prompt_embed, negative_prompt_embeds=negative_embed, image=image).images[0]
            else:
                raise Exception("Invalid combination of parameters for image generation")
        except Exception as e:
            logging.error(f"Error while generating image: {e}\n{traceback.format_exc()}")
            raise e
        finally:
            sys.stderr = original_stderr
        return new_image

    async def generate_image(
        self,
        model_id: int,
        prompt: str,
        side_x: int,
        side_y: int,
        steps: int,
        negative_prompt: str = "",
        img2img: bool = False,
        image: Image = None,
        promptless_variation: bool = False,
        upscaler: bool = False
    ):
        SAG = self.user_config.get("enable_sag", False)
        pipe = await self._prepare_pipe_async(
            model_id,
            img2img,
            promptless_variation,
            SAG,
            upscaler
        )
        self.prompt_manager = self._get_prompt_manager(pipe)

        if SAG and "sag_capable" in self.model_config and self.model_config["sag_capable"] is None or self.model_config["sag_capable"] is False:
            side_x, side_y = ResolutionManager.validate_sag_resolution(self.model_config, self.user_config, side_x, side_y)
            
        # The final cap-off attempt to clamp memory use.
        side_x, side_y = self._get_maximum_generation_res(side_x, side_y)
        new_image = await self._generate_image_with_pipe_async(
            pipe,
            prompt,
            side_x,
            side_y,
            steps,
            negative_prompt,
            self.user_config,
            image,
            promptless_variation,
            upscaler
        )
        # Get the rescaled resolution
        resolution = self._get_rescaled_resolution(self.user_config, side_x, side_y)
        side_x = resolution["width"]
        side_y = resolution["height"]
        logging.info(f"Rescaled resolution: {side_x}x{side_y}")
        new_image = new_image.resize((int(side_x), int(side_y)), Image.ANTIALIAS)

        return new_image
    
    def _get_generator(self, user_config: dict):
        self.seed = user_config.get("seed", None)
        if self.seed is None:
            self.seed = int(time.time())
        generator = torch.manual_seed(self.seed)
        logging.info(f"Seed: {self.seed}")
        return generator

    def _get_prompt_manager(self, pipe):
        logging.debug(f"Initialized the Compel")
        return PromptManipulation(pipeline=pipe, device=self.pipeline_manager.device)
    
    def _get_rescaled_resolution(self, user_config, side_x, side_y):
        resolution = { "width": side_x, "height": side_y }
        return ResolutionManager.nearest_scaled_resolution(resolution, user_config)
    
    def _get_maximum_generation_res(self, side_x, side_y):
        resolution = ResolutionManager.nearest_generation_resolution(side_x, side_y)
        return resolution["width"], resolution["height"]