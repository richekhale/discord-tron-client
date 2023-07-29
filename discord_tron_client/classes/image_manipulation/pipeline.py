import logging, sys, torch, gc, traceback, time, asyncio, diffusers
from torch.cuda import OutOfMemoryError
from tqdm import tqdm
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.image_manipulation.resolution import ResolutionManager
from discord_tron_client.classes.image_manipulation import upscaler as upscaling_helper
from discord_tron_client.classes.image_manipulation.prompt_manipulation import (
    PromptManipulation,
)
from discord_tron_client.classes.tqdm_capture import TqdmCapture
from discord_tron_client.classes.discord_progress_bar import DiscordProgressBar
from discord_tron_client.message.discord import DiscordMessage
from PIL import Image
from discord_tron_client.classes.image_manipulation.metadata import ImageMetadata
from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
    Text2ImgPipelineRunner,
    Img2ImgPipelineRunner,
    SdxlBasePipelineRunner,
    SdxlRefinerPipelineRunner,
    KandinskyTwoTwoPipelineRunner,
    DeepFloydPipelineRunner,
    runner_map,
)

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
        self.prompt_manager = None

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
            guidance_scale = user_config.get("guidance_scaling", 7.5)
            guidance_scale = min(float(guidance_scale), float(20))

            self.gpu_power_consumption = 0.0
            generator = self._get_generator(user_config=user_config)
            # Strip the user_config piece from the prompt.
            prompt = PromptManipulation.remove_duplicate_prompts(prompt, user_config)
            prompt_embed = None
            negative_embed = None
            pooled_embed = None
            negative_pooled_embed = None
            if (
                self.prompt_manager is not None
                and not promptless_variation
                and self.prompt_manager.should_enable(pipe, user_config)
                and self.config.enable_compel()
            ):
                embeddings = self.prompt_manager.process_long_prompt(
                    positive_prompt=prompt, negative_prompt=negative_prompt
                )
                if len(embeddings) == 2:
                    prompt_embed, negative_embed = embeddings
                elif len(embeddings) == 4:
                    (
                        prompt_embed,
                        negative_embed,
                        pooled_embed,
                        negative_pooled_embed,
                    ) = embeddings
                else:
                    raise ValueError(
                        f"Unexpected number of embeddings returned: {len(embeddings)}"
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
                        pooled_embed=pooled_embed,
                        negative_pooled_embed=negative_pooled_embed,
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
        pooled_embed=None,
        image_return_type="pil",
        negative_pooled_embed=None,
    ):
        original_stderr = sys.stderr
        sys.stderr = self.tqdm_capture
        batch_size = self.config.maximum_batch_size()
        try:
            use_latent_result = user_config.get("latent_refiner", True)
            self.pipeline_manager.to_accelerator(pipe)
            denoising_start = None
            preprocessed_images = None
            user_model = user_config.get("model", "")
            if use_latent_result:
                if user_config.get("refiner_strength", 0.5) > 1.0:
                    raise ValueError("refiner_strength must be between 0.0 and 1.0")

                image_return_type = "latent"
                if not type(pipe) in [diffusers.StableDiffusionXLPipeline, diffusers.StableDiffusionXLImg2ImgPipeline]:
                    # We can't send latents directly from a non-SDXL pipeline into the SDXL refiner.
                    image_return_type = "pil"

                # Max inference steps are an inverse relationship of the refiner strength with the base steps.
                denoising_start = 1 - user_config.get("refiner_strength", 0.5)
                logging.debug(
                    f"Final inference step: {denoising_start}, steps: {steps}"
                )
            if not promptless_variation and image is None:
                logging.info(f'Running text2img with batch_size {batch_size} via model {user_model}.')
                # text2img workflow
                if "ptx0/s1" in user_model or "stable-diffusion-xl" in user_model or "sdxl" in user_model:
                    pipeline_runner = runner_map["sdxl_base"](pipeline=pipe)
                elif "ptx0/s2" in user_model or "xl-refiner" in user_model:
                    pipeline_runner = runner_map["sdxl_refiner"](pipeline=pipe)
                elif "kandinsky-2-2" in user_model:
                    pipeline_runner = runner_map["kandinsky_2.2"](decoder=pipe, pipeline_manager=self.pipeline_manager)
                elif "DeepFloyd" in user_model:
                    pipeline_runner = runner_map["deep_floyd"](stage1=pipe, pipeline_manager=self.pipeline_manager, diffusion_manager=self)
                    # DeepFloyd pipeline handles all of this.
                    use_latent_result = False
                    image_return_type = "pil"
                else:
                    pipeline_runner = runner_map["text2img"](pipeline=pipe)
                preprocessed_images = pipeline_runner(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    user_config=user_config,
                    prompt_embeds=prompt_embed,
                    negative_prompt_embeds=negative_embed,
                    pooled_prompt_embeds=pooled_embed,
                    negative_pooled_prompt_embeds=negative_pooled_embed,
                    num_images_per_prompt=batch_size,
                    height=side_y,
                    width=side_x,
                    num_inference_steps=int(float(steps)),
                    denoising_end=denoising_start,
                    guidance_rescale=float(user_config.get("guidance_rescale", 0.3)),
                    guidance_scale=float(guidance_scale),
                    output_type=image_return_type,
                    generator=generator,
                )
                if use_latent_result:
                    logging.info(f'Putting text2img latents into refiner at {denoising_start * 100} percent of the way through the process..')
                    preprocessed_images = self._refiner_pipeline(
                        images=preprocessed_images,
                        user_config=user_config,
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        denoising_start=denoising_start,
                    )
                new_image = self._controlnet_all_images(
                    preprocessed_images=preprocessed_images,
                    user_config=user_config,
                    generator=generator,
                )
            elif not upscaler and not promptless_variation and image is not None:
                logging.info(f'Running img2img with batch_size {batch_size} via model {user_model}.')
                # Img2Img workflow
                guidance_scale = 7.5
                new_image = pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=batch_size,
                    image=image,
                    strength=user_config["strength"],
                    num_inference_steps=user_config.get("steps", 20),
                    denoising_end=0.8 if use_latent_result else None,
                    output_type=image_return_type,
                    guidance_scale=guidance_scale,
                ).images
                if use_latent_result:
                    new_image = self._refiner_pipeline(
                        images=new_image,
                        user_config=user_config,
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        denoising_start=0.8,
                    )
                new_image = self._controlnet_all_images(
                    preprocessed_images=preprocessed_images or new_image,
                    user_config=user_config,
                    generator=generator,
                )
            elif promptless_variation:
                new_image = self._controlnet_pipeline(
                    image=image,
                    user_config=user_config,
                    pipe=pipe,
                    generator=generator,
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    controlnet_strength=user_config.get("tile_strength", 0.3),
                )
            elif upscaler:
                logging.info("Upscaling image using Real-ESRGAN!")
                new_image = self.pipeline_manager.upscale_image(new_image)
            else:
                raise Exception(
                    "Invalid combination of parameters for image generation"
                )
        except OutOfMemoryError as e:
            logging.warn(f"Out of memory error: {e}")
            self.pipeline_manager.delete_pipes()
            raise Exception(
                "The GPU ran out of memory when generating your awesome image. Please try again with a lower size.."
            )
        except Exception as e:
            logging.error(
                f"Error while generating image: {e}\n{traceback.format_exc()}"
            )
            raise e
        finally:
            sys.stderr = original_stderr
            # This should help with sporadic GPU memory errors.
            # https://github.com/damian0815/compel/issues/24
            try:
                del prompt_embed
                del negative_embed
                self.pipeline_manager.to_cpu(pipe)
                gc.collect()
            except Exception as e:
                logging.warn(f"Could not cleanly clear the GC: {e}")

        # Now we upscale using Real-ESRGAN.
        should_upscale = user_config.get("hires_fix", False)
        if should_upscale:
            logging.info("Upscaling image using Real-ESRGAN!")
            new_image = self.pipeline_manager.upscale_image(new_image)
        image_params = {
            "seed": self.seed,
            "guidance_scaling": guidance_scale,
            "strength": user_config.get("strength", 0.5),
        }
        return self._encode_output(
            new_image, positive_prompt, user_config, image_params
        )

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
        if not promptless_variation and "kandinsky" not in model_id:
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
        self.pipeline_manager.clear_cuda_cache()

        return new_image

    def _get_generator(self, user_config: dict, override_seed: int = None):
        if override_seed is None:
            self.seed = user_config.get("seed", None)
        import random

        if self.seed is None or int(self.seed) == 0:
            self.seed = int(time.time())
            self.seed = int(self.seed) + random.randint(-5, 5)
        elif int(self.seed) < 0:
            self.seed = random.randint(0, 2**32)
        generator = torch.Generator(device="cpu")
        go_seed = int(self.seed)
        if override_seed is not None:
            go_seed = int(override_seed)
        generator.manual_seed(int(go_seed))
        logging.info(f"Seed: {go_seed}")
        return generator

    def _get_prompt_manager(
        self, pipe, device="cpu", use_second_encoder_only: bool = False
    ):
        is_gpu = next(pipe.unet.parameters()).is_cuda
        if is_gpu:
            if device == "cpu":
                logging.warning(
                    f"Prompt manager was requested to be placed on the CPU, but the unet is already on the GPU. We have to adjust the prompt manager, to the GPU."
                )
            device = "cuda"
        return PromptManipulation(
            pipeline=pipe,
            device=device,
            use_second_encoder_only=use_second_encoder_only,
        )

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
        img = input_image.resize((W, H), resample=Image.BICUBIC)
        return img

    def _controlnet_pipeline(
        self,
        image: Image,
        user_config: dict,
        pipe,
        generator,
        prompt: str = None,
        negative_prompt: str = None,
        controlnet_strength: float = None
    ):
        # Get the image width/height from 'image' if it's provided
        logging.info(f"Running promptless variation with image.size {image.size}.")
        width, height = image.size
        if width != 1024 and height != 1024:
            # If neither width nor height is 1024, resize the image so that one is, while
            # maintaining the aspect ratio.
            image = self._resize_for_condition_image(input_image=image, resolution=1024)
        if prompt is None:
            prompt = user_config["tile_positive"]
            negative_prompt = user_config["tile_negative"]
        prompt_embed, negative_embed = None, None
        if self.config.enable_compel():
            controlnet_prompt_manager = self._get_prompt_manager(pipe)
            prompt_embed, negative_embed = controlnet_prompt_manager.process_long_prompt(
                positive_prompt=prompt, negative_prompt=negative_prompt
            )
        new_image = pipe(
            prompt_embeds=prompt_embed,
            negative_prompt_embeds=negative_embed,
            image=image,
            controlnet_conditioning_image=image,
            width=image.size[0],
            height=image.size[1],
            strength=controlnet_strength,
            generator=generator,
            num_inference_steps=user_config.get("tile_steps", 32),
        ).images[0]
        return new_image

    def _refiner_pipeline(
        self,
        images: list,
        user_config: dict,
        prompt: str = None,
        negative_prompt: str = None,
        random_seed=False,
        denoising_start=None,
    ):
        # Get the image width/height from 'image' if it's provided
        logging.info(f"Running SDXL Refiner..")
        pipe = self.pipeline_manager.get_sdxl_refiner_pipe()
        pipeline_runner = runner_map["sdxl_refiner"](pipeline=pipe)
        prompt_embed = None
        negative_embed = None
        pooled_embed = None
        negative_pooled_embed = None
        if self.config.enable_compel():
            logging.info(f'SDXL Refiner is using Compel prompt embed weighting.')
            refiner_prompt_manager = self._get_prompt_manager(pipe, use_second_encoder_only=True)
            prompt_embed, negative_embed, pooled_embed, negative_pooled_embed = refiner_prompt_manager.process_long_prompt(
                positive_prompt=prompt, negative_prompt=negative_prompt
            )

        # Reverse the bits in the seed:
        seed_flip = int(self.seed)
        return pipeline_runner(
            generator=torch.Generator(device="cpu").manual_seed(int(seed_flip)),
            prompt_embeds=prompt_embed,
            negative_prompt_embeds=negative_embed,
            pooled_prompt_embeds=pooled_embed,
            negative_pooled_prompt_embeds=negative_pooled_embed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=images,
            user_config=user_config,
            output_type="pil",
            num_images_per_prompt=1,
            guidance_scale=float(user_config.get("refiner_guidance", 7.5)),
            guidance_rescale=float(user_config.get("refiner_guidance_rescale", 0.0)),
            strength=float(user_config.get("refiner_strength", 0.5)),
            aesthetic_score=float(user_config.get("aesthetic_score", 10.0)),
            negative_aesthetic_score=float(
                user_config.get("negative_aesthetic_score", 1.0)
            ),
            num_inference_steps=int(user_config.get("steps", 20)),
            denoising_start=denoising_start,
        )

    def _controlnet_all_images(
        self, preprocessed_images: list, user_config: dict, generator, prompt: str = None, negative_prompt: str = None, controlnet_strength: float = None
    ):
        if controlnet_strength is None:
            controlnet_strength = user_config.get("tile_strength", 0.3)
        logging.debug(f'User ControlNet strength: {controlnet_strength}')
        if float(controlnet_strength) == 0.0:
            # Zero strength = Zero CTU.
            return preprocessed_images

        idx = 0
        controlnet_pipe = self.pipeline_manager.get_controlnet_pipe()
        for image in preprocessed_images:
            preprocessed_images[idx] = self._controlnet_pipeline(
                image=image,
                user_config=user_config,
                pipe=controlnet_pipe,
                generator=generator,
                prompt=prompt,
                negative_prompt=negative_prompt,
                controlnet_strength=controlnet_strength,
            )
            gc.collect()
            idx += 1
        del controlnet_pipe
        return preprocessed_images

    def _encode_image_metadata(self, image: Image, prompt, user_config, image_params):
        attributes = {
            "prompt": prompt,
            "original_user": str(user_config["user_id"]),
            "guidance_scaling": str(image_params.get("guidance_scaling", 7.5)),
            "seed": str(image_params["seed"]),
        }
        if not user_config.get("encode_metadata", True) or not hasattr(
            image, "save"
        ):
            return image
        return ImageMetadata.encode(image, user_config, attributes)

    def _encode_images_metadata(
        self, images: list, prompt, user_config, image_params: dict = {}
    ):
        idx = 0
        for image in images:
            if not hasattr(image, 'save'):
                logging.warning(f'Returning un-processable image: {type(image)}')
                return images
            images[idx] = self._encode_image_metadata(
                image, prompt, user_config, image_params
            )
            idx += 1
        return images

    def _encode_output(self, output, prompt, user_config, image_params: dict = {}):
        if type(output) == list:
            return self._encode_images_metadata(
                output, prompt, user_config, image_params
            )
        return self._encode_image_metadata(output, prompt, user_config, image_params)
