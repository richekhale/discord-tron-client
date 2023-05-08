import logging
import sys
import torch
from torch.cuda import OutOfMemoryError
import traceback
import time
import asyncio
from tqdm import tqdm
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.image_manipulation.resolution import ResolutionManager
from discord_tron_client.classes.image_manipulation.prompt_manipulation import (
    PromptManipulation,
)
from discord_tron_client.classes.tqdm_capture import TqdmCapture
from discord_tron_client.classes.discord_progress_bar import DiscordProgressBar
from discord_tron_client.message.discord import DiscordMessage
from PIL import Image

hardware = HardwareInfo()

class PipelineRunner:
    def __init__(
        self,
        model_manager,
        pipeline_manager,
        app_config: AppConfig,
        user_config: dict,
        discord_msg,
        websocket,
        model_config: dict = {},
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
            websocket=websocket, context=discord_msg, module_command="edit"
        )
        # An object to manage a progress bar for Discord.
        self.progress_bar = DiscordProgressBar(
            websocket=websocket,
            websocket_message=self.progress_bar_message,
            progress_bar_steps=100,
            progress_bar_length=20,
            discord_first_message=discord_msg,
        )
        self.tqdm_capture = TqdmCapture(self.progress_bar, main_loop)
        self.websocket = websocket
        self.model_config = model_config

    async def _prepare_pipe_async(
        self,
        user_config: dict,
        scheduler_config: dict,
        resolution,
        model_id: int,
        variation: bool = False,
        promptless_variation: bool = False,
        upscaler: bool = False,
    ):
        loop = asyncio.get_event_loop()
        loop_return = await loop.run_in_executor(
            AppConfig.get_image_worker_thread(),  # Use a dedicated image processing thread worker.
            self._prepare_pipe,
            user_config,
            scheduler_config,
            resolution,
            model_id,
            variation,
            promptless_variation,
            upscaler,
        )
        return loop_return

    def _prepare_pipe(
        self,
        user_config: dict,
        scheduler_config: dict,
        resolution: dict,
        model_id: int,
        variation: bool = False,
        promptless_variation: bool = False,
        upscaler: bool = False,
    ):
        logging.info(f"Retrieving pipe for model {model_id}")
        pipe = self.pipeline_manager.get_pipe(
            user_config,
            scheduler_config,
            resolution,
            model_id,
            prompt_variation=variation,
            promptless_variation=promptless_variation,
            upscaler=upscaler,
        )
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
            upscaler,
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
        upscaler: bool = False,
    ):
        try:
            guidance_scale = user_config.get("guidance_scale", 7.5)
            guidance_scale = min(guidance_scale, 20)

            self.gpu_power_consumption = 0.0
            generator = self._get_generator(user_config=user_config)

            prompt_embed = None
            negative_embed = None
            if not promptless_variation:
                prompt_embed, negative_embed = self.prompt_manager.process_long_prompt(
                    positive_prompt=prompt, negative_prompt=negative_prompt
                )

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
                        user_config,
                        image,
                        promptless_variation,
                        upscaler,
                        positive_prompt=prompt,
                        negative_prompt=negative_prompt,
                    )
            self.gpu_power_consumption = self.tqdm_capture.gpu_power_consumption
            return new_image
        except Exception as e:
            logging.error(
                f"Error while generating image: {e}\n{traceback.format_exc()}"
            )
            raise e

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
        user_config: dict,
        image: Image = None,
        promptless_variation: bool = False,
        upscaler: bool = False,
        positive_prompt="",
        negative_prompt="",
    ):
        original_stderr = sys.stderr
        sys.stderr = self.tqdm_capture
        try:
            alt_weight_algorithm = user_config.get("alt_weight_algorithm", False)
            if not promptless_variation and image is None:
                if not alt_weight_algorithm:
                    # Default "long prompt weighting" pipeline
                    new_image = pipe(
                        use_karras_sigmas=True,
                        prompt=positive_prompt,
                        num_images_per_prompt=4,
                        height=side_y,
                        width=side_x,
                        num_inference_steps=int(float(steps)),
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images
                else:
                    # Use the Compel library's prompt weights as input instead.
                    new_image = pipe(
                        use_karras_sigmas=True,
                        positive_embeds=prompt_embed,
                        num_images_per_prompt=4,
                        height=side_y,
                        width=side_x,
                        num_inference_steps=int(float(steps)),
                        negative_embeds=negative_embed,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images
            elif not upscaler and not promptless_variation and image is not None:
                if not alt_weight_algorithm:
                    new_image = pipe.img2img(
                        prompt=positive_prompt,
                        num_images_per_prompt=4,
                        image=image,
                        strength=user_config["strength"],
                        num_inference_steps=int(float(steps)),
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images
                else:
                    new_image = pipe(
                        prompt_embeds=prompt_embed,
                        num_images_per_prompt=4,
                        image=image,
                        strength=user_config["strength"],
                        num_inference_steps=int(float(steps)),
                        negative_prompt_embeds=negative_embed,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images
            elif promptless_variation:
                # Get the image width/height from 'image' if it's provided
                logging.info(
                    f"Running promptless variation with image.size {image.size}."
                )
                if image is not None:
                    side_x = image.width
                    side_y = image.height
                    # side_x, side_y = ResolutionManager.nearest_generation_resolution(
                    #     side_x, side_y
                    # )
                image = self._resize_for_condition_image(input_image=image, resolution=1024)
                new_image = pipe(
                    prompt=user_config["tile_positive"],
                    negative_prompt=user_config["tile_negative"],
                    image=image,
                    controlnet_conditioning_image=image,
                    width=image.size[0],
                    height=image.size[1],
                    strength=user_config["strength"],
                    generator=generator,
                    num_inference_steps=int(float(steps)),
                ).images[0]
            elif upscaler:
                new_image = pipe(
                    prompt_embeds=prompt_embed,
                    num_images_per_prompt=4,
                    negative_prompt_embeds=negative_embed,
                    image=image,
                    num_inference_steps=int(float(steps)),
                ).images
            else:
                raise Exception(
                    "Invalid combination of parameters for image generation"
                )
        except OutOfMemoryError as e:
            logging.warn(f"Out of memory error: {e}")
            self.pipeline_manager.delete_pipes()
            raise Exception("The GPU ran out of memory when generating your awesome image. Please try again with a lower size..")
        except Exception as e:
            logging.error(
                f"Error while generating image: {e}\n{traceback.format_exc()}"
            )
            raise e
        finally:
            sys.stderr = original_stderr
        return new_image

    async def generate_image(
        self,
        model_id: int,
        user_config: dict,
        scheduler_config: dict,
        prompt: str,
        side_x: int,
        side_y: int,
        steps: int,
        negative_prompt: str = "",
        image: Image = None,
        prompt_variation: bool = False,
        promptless_variation: bool = False,
        upscaler: bool = False,
    ):
        resolution = {"width": side_x, "height": side_y}
        pipe = await self._prepare_pipe_async(
            user_config,
            scheduler_config,
            resolution,
            model_id,
            prompt_variation,
            promptless_variation,
            upscaler,
        )
        if not promptless_variation:
            self.prompt_manager = self._get_prompt_manager(pipe)

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
            upscaler,
        )
        # Get the rescaled resolution
        resolution = self._get_rescaled_resolution(self.user_config, side_x, side_y)
        side_x = resolution["width"]
        side_y = resolution["height"]
        logging.info(f"Rescaled resolution: {side_x}x{side_y}")
        if isinstance(new_image, list) and not promptless_variation:
            for i in range(len(new_image)):
                new_image[i] = new_image[i].resize(
                    (int(side_x), int(side_y)), resample=Image.LANCZOS
                )
        if hasattr(new_image, "resize") and not promptless_variation:
            new_image = new_image.resize(
                (int(side_x), int(side_y)), resample=Image.LANCZOS
            )

        self.pipeline_manager.clear_cuda_cache()

        return new_image

    def _get_generator(self, user_config: dict):
        self.seed = user_config.get("seed", None)
        import random

        if self.seed is None or int(self.seed) == 0:
            self.seed = int(time.time())
            self.seed = int(self.seed) + random.randint(-5, 5)
        elif int(self.seed) < 0:
            self.seed = random.randint(0, 2**32)
        generator = torch.Generator(device=self.pipeline_manager.device)
        generator.manual_seed(int(self.seed))
        logging.info(f"Seed: {self.seed}")
        return generator

    def _get_prompt_manager(self, pipe):
        logging.debug(f"Initialized the Compel")
        return PromptManipulation(pipeline=pipe, device=self.pipeline_manager.device)

    def _get_rescaled_resolution(self, user_config, side_x, side_y):
        resolution = {"width": side_x, "height": side_y}
        return ResolutionManager.nearest_scaled_resolution(resolution, user_config)

    def _get_maximum_generation_res(self, side_x, side_y):
        return ResolutionManager.nearest_generation_resolution(side_x, side_y)

    def _resize_for_condition_image(self, input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img