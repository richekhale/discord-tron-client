import logging, sys, torch, traceback, time, asyncio
from tqdm import tqdm
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.image_manipulation.resolution import ResolutionManager
from discord_tron_client.classes.tqdm_capture import TqdmCapture
from discord_tron_client.classes.discord_progress_bar import DiscordProgressBar
from discord_tron_client.message.discord import DiscordMessage
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

class PipelineRunner:
    def __init__(self, model_manager, pipeline_manager, app_config, user_config, discord_msg, websocket):
        # General AppConfig() object access.
        self.config = app_config
        main_loop = asyncio.get_event_loop()
        if main_loop is None:
            raise Exception("AppConfig.main_loop is not set!")
        # The received user_config item from TRON master.
        self.user_config = user_config
        # Managers.
        self.model_manager = model_manager
        self.pipeline_manager = pipeline_manager
        # A message template for the WebSocket events.
        self.progress_bar_message = DiscordMessage(websocket=websocket, context=discord_msg, module_command="edit")
        # An object to manage a progress bar for Discord.
        self.progress_bar = DiscordProgressBar(websocket=websocket, websocket_message=self.progress_bar_message, progress_bar_steps=100, progress_bar_length=20, discord_first_message=discord_msg)
        self.tqdm_capture = TqdmCapture(self.progress_bar, main_loop)
        self.websocket = websocket

    async def _prepare_pipe_async(self, model_id, img2img: bool = False, promptless_variation: bool = False):
        loop = asyncio.get_event_loop()
        loop_return = await loop.run_in_executor(
            None, # Use the default ThreadPoolExecutor
            self._prepare_pipe,
            model_id,
            img2img,
            promptless_variation
        )
        return loop_return


    def _prepare_pipe(self, model_id, img2img: bool = False, promptless_variation: bool = False):
        logging.info("Retrieving pipe for model " + str(model_id))
        if not promptless_variation:
            pipe = self.pipeline_manager.get_pipe(model_id, img2img)
        else:
            pipe = self.pipeline_manager.get_variation_pipe(model_id)
        logging.info("Copied pipe to the local context")
        return pipe

    async def _generate_image_with_pipe_async(self, pipe, prompt, side_x, side_y, steps, negative_prompt, user_config, image: Image = None, promptless_variation: bool = False):
        loop = asyncio.get_event_loop()
        loop_return = await loop.run_in_executor(
            AppConfig.get_image_worker_thread(), # Use a dedicated image processing thread worker.
            self._generate_image_with_pipe,
            pipe,
            prompt,
            side_x,
            side_y,
            steps,
            negative_prompt,
            user_config,
            image,
            promptless_variation
        )
        delete_progress_bar = DiscordMessage(websocket=self.websocket, context=self.progress_bar_message.context, module_command="delete")
        for attempt in range(1, 6):
            if not self.websocket or not hasattr(self.websocket, "open") or self.websocket.open != True:
                logging.warn("WebSocket connection is not open. Retrieving fresh instance.")
                self.websocket = AppConfig.get_websocket()
                await asyncio.sleep(2)
            else:
                logging.debug("WebSocket connection is open. Continuing.")
                break
        await self.websocket.send(delete_progress_bar.to_json())
        return loop_return

    def _generate_image_with_pipe(self, pipe, prompt, side_x, side_y, steps, negative_prompt, user_config, image: Image = None, promptless_variation: bool = False):
        try:
            with torch.no_grad():
                with tqdm(total=steps, ncols=100, file=self.tqdm_capture) as pbar:
                    if not promptless_variation and image is None:
                        # We're not doing a promptless variation, and we don't have an image to start with.
                        new_image = pipe(
                            prompt=prompt,
                            height=side_y,
                            width=side_x,
                            num_inference_steps=int(float(steps)),
                            negative_prompt=negative_prompt,
                        ).images[0]
                    elif image is not None:
                        # We have an image to start with. Currently, promptless_variation falls through here. But it has its own pipeline to use.
                        logging.info(f"Image is not None, using it as a starting point for the image generation process: {image}")
                        new_image = pipe(
                            prompt=prompt,
                            image=image,
                            strength=user_config["strength"], # How random the img2img should be. Higher = less.
                            num_inference_steps=int(float(steps)),
                            negative_prompt=negative_prompt,
                        ).images[0]                    
            return new_image
        except Exception as e:
            logging.error("Error while generating image: " + str(e) + " " + str(traceback.format_exc()))

    async def generate_image( self, prompt, model_id, resolution, negative_prompt, steps, positive_prompt, user_config, image: Image = None, promptless_variation: bool = False):
        logging.info("Initializing image generation pipeline...") 
        use_attention_scaling, steps = self.check_attention_scaling(resolution, steps)
        aspect_ratio, side_x, side_y = ResolutionManager.get_aspect_ratio_and_sides(self.config, resolution)
        img2img = False
        if image is not None:
            img2img = True
        pipe = await self._prepare_pipe_async(model_id, img2img, use_attention_scaling)
        logging.info("REDIRECTING THE PRECIOUS, STDOUT... SORRY IF THAT UPSETS YOU")

        original_stderr = sys.stderr
        sys.stderr = self.tqdm_capture

        entire_prompt = self.combine_prompts(prompt, positive_prompt)

        for attempt in range(1, 1):
            try:
                logging.info(f"Attempt {attempt}: Generating image...")
                image = await self._generate_image_with_pipe_async(pipe, entire_prompt, side_x, side_y, steps, negative_prompt, user_config, image, promptless_variation)
                logging.info("Image generation successful!")
                break
            except Exception as e:
                logging.error(f"Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}")
                if attempt < 5:
                    time.sleep(5)
                else:
                    raise RuntimeError("Maximum retries reached, image generation failed")
            finally:
                sys.stderr = original_stderr
        try:
            scaling_target = ResolutionManager.nearest_scaled_resolution(resolution, user_config, self.config.get_max_resolution_by_aspect_ratio(aspect_ratio))
            if scaling_target != resolution:
                logging.info("Rescaling image to nearest resolution...")
                image = image.resize((scaling_target["width"], scaling_target["height"]))
            return image
        except Exception as e:
            logging.error(f"Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}")
            raise RuntimeError("Error resizing image: {e}")


    def check_attention_scaling(self, resolution, steps):
        is_attn_enabled = self.config.get_attention_scaling_status()
        use_attention_scaling = False
        if resolution is not None and is_attn_enabled:
            scaling_factor = self.get_scaling_factor(
                resolution["width"], resolution["height"], self.resolutions
            )
            logging.info(
                f"Scaling factor for {resolution['width']}x{resolution['height']}: {scaling_factor}"
            )
            if scaling_factor < 50:
                logging.info(
                    "Resolution "
                    + str(resolution["width"])
                    + "x"
                    + str(resolution["height"])
                    + " has a pixel count greater than threshold. Using attention scaling expects to take 30 seconds."
                )
                use_attention_scaling = True
                if steps > scaling_factor:
                    steps = scaling_factor
        return use_attention_scaling, steps

    def combine_prompts(self, prompt, positive_prompt):
        entire_prompt = prompt
        if positive_prompt is not None:
            entire_prompt = str(prompt) + " , " + str(positive_prompt)
        return entire_prompt
