from discord_tron_client.classes.image_manipulation.pipeline_runners.base_runner import (
    BasePipelineRunner,
)
from diffusers import (
    DiffusionPipeline,
    IFPipeline,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image
import logging, random, torch
from typing import Union, List, Optional
from transformers import T5EncoderModel

from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.hardware import HardwareInfo

hardware_info = HardwareInfo()
config = AppConfig()

scheduler_map = {
    "multistep": DPMSolverMultistepScheduler,
    "ddpm": DDPMScheduler,
}


@torch.no_grad()
def encode_prompt_with_max_seq_len(
    self,
    prompt: Union[str, List[str]],
    do_classifier_free_guidance: bool = True,
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    clean_caption: bool = False,
    max_sequence_len: int = 512,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            whether to use classifier free guidance or not
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            number of images that should be generated per prompt
        device: (`torch.device`, *optional*):
            torch device to place the resulting embeddings on
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
            Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        clean_caption (bool, defaults to `False`):
            If `True`, the function will preprocess and clean the provided caption before encoding.
    """
    if prompt is not None and negative_prompt is not None:
        if type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )

    if device is None:
        device = self._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
    max_length = max_sequence_len

    if prompt_embeds is None:
        prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, max_length - 1 : -1]
            )
            logging.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )

        attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

    if self.text_encoder is not None:
        dtype = self.text_encoder.dtype
    elif self.unet is not None:
        dtype = self.unet.dtype
    else:
        dtype = None

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_tokens = self._text_preprocessing(
            uncond_tokens, clean_caption=clean_caption
        )
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        attention_mask = uncond_input.attention_mask.to(device)

        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(
            1, num_images_per_prompt, 1
        )
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
    else:
        negative_prompt_embeds = None

    return prompt_embeds, negative_prompt_embeds


class DeepFloydPipelineRunner(BasePipelineRunner):
    def __init__(self, stage1, pipeline_manager, diffusion_manager):
        super().__init__(
            pipeline=None,
            pipeline_manager=pipeline_manager,
            diffusion_manager=diffusion_manager,
        )
        self.stage1 = stage1  # DeepFloyd/IF-I-XL-v1.0
        self.stage1_fused = False
        self.stage1_should_fuse = True
        self.stage1_lora_scale = 0.25
        self.stage2 = None  # DeepFloyd/IF-II-L-v1.0
        self.stage3 = None  # Upscaler
        self.safety_modules = {
            "feature_extractor": self.stage1.feature_extractor,
            "safety_checker": None,
            "watermarker": None,
        }

    def _invoke_sdxl(
        self, user_config: dict, prompt: str, negative_prompt: str, images: Image
    ):
        logging.debug(f"Upscaling DeepFloyd output using SDXL refiner.")
        # Upscale using PIL, by 4:
        if type(images) != list:
            images = [images]
        idx = 0
        for image in images:
            if hasattr(image, "width"):
                width = image.width * 4
                height = image.height * 4
                logging.debug(
                    f"_invoke_sdxl resizing image from {image.width}x{image.height} to {width}x{height}."
                )
                images[idx] = image.resize((width, height), Image.LANCZOS)
            else:
                logging.debug(f"_invoke_sdxl not resizing non-Image inputs.")
                break
            idx += 1
        logging.debug(f"Generating SDXL-refined DeepFloyd output.")
        output = self.diffusion_manager._refiner_pipeline(
            images=images,
            user_config=user_config,
            prompt=prompt,
            negative_prompt=negative_prompt,
            random_seed=False,
            denoising_start=None,
        )
        logging.debug(f"Generating SDXL-refined DeepFloyd output has completed.")
        self._cleanup_pipes()
        return output

    def _setup_stage2(self, user_config):
        stage2_model = "DeepFloyd/IF-II-M-v1.0"
        logging.debug(f"Configuring DF-IF Stage II Pipeline: {stage2_model}")
        if self.stage2 is not None:
            logging.info(
                f"Keeping existing {stage2_model} model with {type(self.stage2)} pipeline."
            )
            return
        logging.debug(f"Retrieving DeepFloyd Stage II pipeline.")
        self.stage2 = self.pipeline_manager.get_pipe(
            model_id=stage2_model, user_config=user_config, custom_text_encoder=-1
        )
        logging.debug(f"Retrieving DeepFloyd Stage II pipeline has completed.")

    def _invoke_stage2(
        self,
        image: Image,
        user_config,
        prompt_embeds,
        negative_embeds,
        generators,
        width=64,
        height=64,
        output_type="pt",
    ):
        self._setup_stage2(user_config)
        s2_width = width * 4
        s2_height = height * 4
        logging.debug(f"Generating DeepFloyd Stage2 output at {s2_width}x{s2_height}.")
        stage2_result = self.stage2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=max(
                1,
                min(
                    50,
                    int(
                        self.parameters.get(
                            "steps_2", user_config.get("df_inference_steps_2", 20)
                        )
                    ),
                ),
            ),
            output_type=output_type,
            width=s2_width,
            height=s2_height,
            num_images_per_prompt=1,
            guidance_scale=max(
                0,
                min(
                    20,
                    float(
                        self.parameters.get(
                            "df_guidance_scale_2",
                            user_config.get("df_guidance_scale_2", 5.7),
                        )
                    ),
                ),
            ),
            generator=generators,
        ).images
        logging.debug(f"Generating DeepFloyd Stage2 output has completed.")
        self._cleanup_pipes()
        return stage2_result

    def _setup_stage3(self, user_config):
        stage3_model = "stabilityai/stable-diffusion-x4-upscaler"
        if self.stage3 is not None:
            logging.info(f"Keeping existing {stage3_model} model.")
            return
        logging.debug(f"Retrieving DeepFloyd Stage III pipeline.")
        self.stage3 = self.pipeline_manager.get_pipe(
            model_id=stage3_model,
            user_config=user_config,
            safety_modules=self.safety_modules,
        )
        logging.debug(f"Retrieving DeepFloyd Stage III pipeline has completed.")
        return

    def _invoke_stage3(
        self,
        prompt: str,
        negative_prompt: str,
        image: Image,
        user_config: dict,
        output_type: str = "pil",
    ):
        self._setup_stage3(user_config)
        user_strength = self.parameters.get(
            "df_stage3_strength", user_config.get("df_stage3_strength", 1.0)
        )
        logging.debug(f"Generating DeepFloyd Stage3 output.")
        output = self.stage3(
            prompt=[prompt] * len(image),
            negative_prompt=[negative_prompt] * len(image),
            image=image,
            noise_level=(100 * user_strength),
            guidance_scale=self.parameters.get(
                "df_guidance_scale_3", user_config.get("df_guidance_scale_3", 5.6)
            ),
            output_type=output_type,
        ).images
        logging.debug(f"Generating DeepFloyd Stage3 output has completed.")
        self._cleanup_pipes()
        return output

    def _invoke_stage1(
        self,
        prompt_embed,
        negative_prompt_embed,
        user_config: dict,
        generators,
        width=64,
        height=64,
    ):
        df_guidance_scale = float(
            self.parameters.get(
                "df_guidance_scale_1", user_config.get("df_guidance_scale_1", 7.2)
            )
        )
        logging.debug(
            f"Generating DeepFloyd Stage1 output at {width}x{height} and {df_guidance_scale} CFG."
        )
        deepfloyd_stage1_lora_model = self.parameters.get(
            "lora", config.get_config_value("deepfloyd_stage1_lora_model", None)
        )
        cross_attention_kwargs = None
        if (
            deepfloyd_stage1_lora_model is not None
            and not self.stage1_fused
            and self.stage1_should_fuse
        ):
            deepfloyd_stage1_lora_model_path = config.get_config_value(
                "deepfloyd_stage1_lora_model_path", "pytorch_lora_weights.safetensors"
            )
            logging.debug(
                f"Loading DeepFloyd Stage1 Lora model from {deepfloyd_stage1_lora_model_path}"
            )
            self.stage1.load_lora_weights(
                deepfloyd_stage1_lora_model,
                weight_name=deepfloyd_stage1_lora_model_path,
            )
            self.stage1_fused = True
            cross_attention_kwargs = {"scale": self.stage1_lora_scale}

        output = self.stage1(
            prompt_embeds=prompt_embed,
            negative_prompt_embeds=negative_prompt_embed,
            num_inference_steps=max(
                1,
                min(
                    100,
                    int(
                        self.parameters.get(
                            "steps_1", user_config.get("df_inference_steps_1", 30)
                        )
                    ),
                ),
            ),
            generator=generators,
            guidance_scale=df_guidance_scale,
            output_type="pt",
            width=width,
            height=height,
            num_images_per_prompt=1,
            cross_attention_kwargs=cross_attention_kwargs,
        ).images

        if self.stage1_fused:
            logging.debug(f"Unloading DeepFloyd Stage1 Lora model")
            try:
                self.stage1.unload_lora_weights()
            except Exception as e:
                logging.warning(f"Possible error unloading DeepFloyd stage I LoRA: {e}")
                self.stage1 = None
            self.stage1_fused = False

        logging.debug(f"Generating DeepFloyd Stage1 output has completed.")
        self._cleanup_pipes()
        return output

    def _setup_text_encoder(self):
        if self.stage1.text_encoder is not None:
            return
        model_id = "DeepFloyd/IF-I-XL-v1.0"
        self.stage1.text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            device_map="auto",
            load_in_8bit=False,
            variant="fp16",
            torch_dtype=self.diffusion_manager.torch_dtype,
        )

    def _embeds(self, prompt: str, negative_prompt: str):
        # DeepFloyd stage 1 can use a more efficient text encoder config.
        prompt, prompt_parameters = self._extract_parameters(prompt)
        if "nolora" in prompt_parameters:
            self.stage1_should_fuse = False
        else:
            self.stage1_should_fuse = True
        if "lora_scale" in prompt_parameters:
            self.stage1_lora_scale = float(prompt_parameters["lora_scale"])
        logging.debug(f"Configuring DeepFloyd text encoder via stage1 pipeline.")
        self._setup_text_encoder()
        logging.debug(f"Generating DeepFloyd text embeds, using stage1 text_encoder.")
        self.max_sequence_len = 512
        if "max_sequence_len" in prompt_parameters:
            self.max_sequence_len = max(
                77, min(512, int(prompt_parameters["max_sequence_len"]))
            )
        self.stage1.encode_prompt = encode_prompt_with_max_seq_len.__get__(
            self.stage1, IFPipeline
        )
        embeds = self.stage1.encode_prompt(
            prompt,
            negative_prompt,
            max_sequence_len=self.max_sequence_len,
            device=self.pipeline_manager.device,
        )
        logging.debug(f"Generating DeepFloyd text embeds has completed.")
        self.stage1.scheduler = scheduler_map[
            prompt_parameters.get("scheduler", "ddpm")
        ].from_config(
            self.stage1.scheduler.config,
            timestep_spacing=prompt_parameters.get("timestep_spacing", "trailing"),
            dynamic_thresholding_ratio=prompt_parameters.get(
                "dynamic_thresholding_ratio", 0.95
            ),
            sample_max_value=prompt_parameters.get("sample_max_value", 1.5),
            steps_offset=prompt_parameters.get("steps_offset", 0),
            thresholding=prompt_parameters.get("thresholding", False),
            variance_type=prompt_parameters.get("variance_type", "learned_range"),
        )
        if self.should_offload():
            # Clean up the text encoder to save VRAM.
            logging.info(f"Clearing up the DeepFloyd text encoder to save VRAM.")
            self.stage1.text_encoder = None
            self.clear_cuda_cache()
        return embeds, prompt_parameters

    def _get_stage1_resolution(self, user_config: dict):
        # Grab the aspect ratio of the user_config['resolution']['width']xuser_config['resolution']['height'],
        # and then use that to ensure that the smaller side is 64px, while the larger side is 64px * aspect_ratio.
        # This has to support portrait or landscape, as well as square images.
        width = user_config.get("resolution", {}).get("width", 1024)
        height = user_config.get("resolution", {}).get("height", 1024)
        logging.debug(
            f"DeepFloyd stage 1 resolution before adjustment is {width}x{height}"
        )
        aspect_ratio = width / height
        # Portrait
        if width < height:
            width = 64
            height = 64 / aspect_ratio
        else:
            height = 64
            width = 64 * aspect_ratio

        # Ensure both dimensions are multiples of 8
        height = (height // 8) * 8
        width = (width // 8) * 8
        logging.debug(
            f"DeepFloyd stage 1 resolution after adjustment is {width}x{height}"
        )

        return int(width), int(height)

    def _get_generators(self, user_config: dict):
        # Create four generators with a seed based on user_config['seed']. Increment for each generator.
        generators = []
        seed = int(user_config.get("seed", 0))
        if int(seed) <= 0:
            seed = random.randint(0, 42042042042)
        for i in range(self.batch_size()):
            generators.append(
                self.diffusion_manager._get_generator(
                    user_config, override_seed=int(seed) + i
                )
            )

        return generators

    def __call__(self, **args):
        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]

        # Grab prompt embeds from T5.
        prompt = args.get("prompt", "")
        negative_prompt = user_config.get("negative_prompt", "")
        logging.debug(f"Positive prompt: {prompt}")
        logging.debug(f"Negative prompt: {negative_prompt}")
        embeds, parameters = self._embeds(
            [prompt] * self.batch_size(), [negative_prompt] * self.batch_size()
        )
        prompt_embeds, negative_embeds = embeds
        generators = self._get_generators(user_config)
        self.parameters = parameters
        try:
            logging.debug(f"Generating stage 1 output.")
            logging.debug(
                f"Shapes of embeds: {prompt_embeds.shape}, {negative_embeds.shape}"
            )
            width, height = self._get_stage1_resolution(user_config)
            stage1_output = self._invoke_stage1(
                prompt_embed=prompt_embeds,
                negative_prompt_embed=negative_embeds,
                width=width,
                height=height,
                user_config=user_config,
                generators=generators,
            )
            logging.debug(f"Generating DeepFloyd Stage2 output.")
            stage2_output = self._invoke_stage2(
                image=stage1_output,
                user_config=user_config,
                prompt_embeds=prompt_embeds,
                negative_embeds=negative_embeds,
                generators=generators,
                width=width,
                height=height,
                output_type=(
                    "pil" if not user_config.get("df_x4_upscaler", True) else "pt"
                ),
            )
            stage3_output = None
            df_x4_upscaler = user_config.get("df_x4_upscaler", True)
            if df_x4_upscaler:
                logging.debug(f"Generating DeepFloyd Stage3 output using x4 upscaler.")
                stage3_output = self._invoke_stage3(
                    prompt=args.get("prompt", ""),
                    negative_prompt=args.get("negative_prompt", ""),
                    image=stage2_output,
                    user_config=user_config,
                )
            df_latent_refiner = user_config.get("df_latent_refiner", False)
            if df_latent_refiner:
                logging.debug(
                    f"Generating DeepFloyd Stage3 output using latent refiner."
                )
                stage3_output = self._invoke_sdxl(
                    images=stage2_output,
                    user_config=user_config,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
            df_esrgan_upscaler = user_config.get("df_esrgan_upscaler", False)
            if df_esrgan_upscaler:
                stage3_output = self.pipeline_manager.upscale_image(
                    stage3_output or stage2_output or stage1_output
                )
            df_controlnet_upscaler = user_config.get("df_controlnet_upscaler", False)
            if df_controlnet_upscaler:
                stage3_output = self.diffusion_manager._controlnet_all_images(
                    preprocessed_images=stage3_output or stage2_output or stage1_output,
                    user_config=user_config,
                    generator=None,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    controlnet_strength=user_config.get("df_controlnet_strength", 1.0),
                )
            if stage3_output is None:
                return stage2_output
            return stage3_output
        except Exception as e:
            logging.error(
                f"DeepFloyd pipeline failed: {e}, traceback: {e.__traceback__}"
            )
            raise e
