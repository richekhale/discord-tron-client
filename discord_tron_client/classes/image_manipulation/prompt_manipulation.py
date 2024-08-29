from discord_tron_client.classes.app_config import AppConfig

import logging

config = AppConfig()
if config.enable_compel():
    from compel import Compel, ReturnedEmbeddingsType
prompt_styles = {
    "base": [
      "{prompt}",  
      "{prompt} striated lines, canvas pattern, blurry"  
    ],
    "alec-baldwin": [
        "the alec baldwin version of {prompt}, absurd, zany, mash-up, shot on the Rust movie set",
        "freedom, actor, movie, professional"
    ],
    "typography": [
        "typography {prompt}, font, typeface, graphic design, centered composition",
        "deformed, misspelt, glitch, noisy, realistic"
    ],
    "wes-anderson": [
        "photo of a cinematic scene, {prompt} shot in the style of wes anderson on 70mm film",
        "comic, newspaper, deformed, glitch, noisy, realistic, stock photo, canvas pattern"
    ],
    "deep-ocean": [
        "underwater photograph of a deep ocean scene, {prompt} in the neon midnight zone",
        "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured, terrestrial, sky"
    ],
    "phone-camera": [
        "iphone samsung galaxy camera photo of {prompt}",
        "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    ],
    "enhance": [
        "breathtaking {prompt} award-winning, professional, highly detailed",
        "ugly, deformed, noisy, blurry, distorted, grainy"
    ],
    "anime": [
        "anime artwork {prompt} anime style, key visual, vibrant, studio anime,  highly detailed",
        "photo, deformed, black and white, realism, disfigured, low contrast"
    ],
    "photographic": [
        "cinematic photo {prompt} 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"
    ],
    "digital-art": [
        "concept art {prompt} digital artwork, illustrative, painterly, matte painting, highly detailed",
        "photo, photorealistic, realism, ugly"
    ],
    "comic-book": [
        "comic {prompt} graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
        "photograph, deformed, glitch, noisy, realistic, stock photo"
    ],
    "fantasy-art": [
        "ethereal fantasy concept art of {prompt} magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white"
    ],
    "analog-film": [
        "analog film photo vintage, detailed Kodachrome, found footage, 1980s {prompt}",
        "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    ],
    "neonpunk": [
        "neonpunk style {prompt} cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    ],
    "isometric": [
        "isometric style {prompt} vibrant, beautiful, crisp, detailed, ultra detailed, intricate",
        "deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic"
    ],
    "lowpoly": [
        "low-poly style {prompt} low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
        "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
    ],
    "origami": [
        "origami style {prompt} paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
        "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
    ],
    "line-art": [
        "line art drawing {prompt} professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic"
    ],
    "craft-clay": [
        "play-doh style {prompt} sculpture, clay art, centered composition, Claymation",
        "sloppy, messy, grainy, highly detailed, ultra textured, photo"
    ],
    "cinematic": [
        "cinematic film still {prompt} shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    ],
    "3d-model": [
        "professional 3d model {prompt} octane render, highly detailed, volumetric, dramatic lighting",
        "ugly, deformed, noisy, low poly, blurry, painting"
    ],
    "pixel-art": [
        "pixel-art {prompt} low-res, blocky, pixel art style, 8-bit graphics",
        "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"
    ],
    "texture": [
        "texture {prompt} top down close-up",
        "ugly, deformed, noisy, blurry"
    ]
}

# Manipulating prompts for the pipeline.
class PromptManipulation:
    def __init__(self, pipeline, device, use_second_encoder_only: bool = False):
        if not config.enable_compel():
            return
        self.is_valid_pipeline(pipeline)
        self.pipeline = pipeline
        if (self.has_dual_text_encoders(pipeline) and not use_second_encoder_only):
            # SDXL Refiner and Base can both use the 2nd tokenizer/encoder.
            logging.debug(f'Initialising Compel prompt manager with dual encoders.')
            self.compel = Compel(
                tokenizer=[
                    self.pipeline.tokenizer,
                    self.pipeline.tokenizer_2
                ],
                text_encoder=[
                    self.pipeline.text_encoder,
                    self.pipeline.text_encoder_2
                ],
                truncate_long_prompts=False,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[
                    False,  # CLIP-L does not produce pooled embeds.
                    True    # CLIP-G produces pooled embeds.
                ]
            )
        elif (self.has_dual_text_encoders(pipeline) and use_second_encoder_only):
            # SDXL Refiner has ONLY the 2nd tokenizer/encoder, which needs to be the only one in Compel.
            logging.debug(f'Initialising Compel prompt manager with just the 2nd text encoder.')
            self.compel = Compel(
                tokenizer=self.pipeline.tokenizer_2,
                text_encoder=self.pipeline.text_encoder_2,
                truncate_long_prompts=False,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=True
            )
        else:
            # Any other pipeline uses the first tokenizer/encoder.
            logging.debug(f'Initialising the Compel prompt manager with a single text encoder.')
            pipe_tokenizer = self.pipeline.tokenizer
            pipe_text_encoder = self.pipeline.text_encoder
            self.compel = Compel(
                tokenizer=pipe_tokenizer,
                text_encoder=pipe_text_encoder,
                truncate_long_prompts=False,
                returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
            )
    def should_enable(self, pipeline, user_config: dict = None):
        if (type(pipeline).__name__ == "StableDiffusionXLPipeline") or (type(pipeline).__name__ == "StableDiffusionPipeline"):
            return True
        return False

    def has_dual_text_encoders(self, pipeline):
        return hasattr(pipeline, "text_encoder_2")

    def is_sdxl_refiner(self, pipeline):
        # SDXL Refiner has the 2nd text encoder, only.
        if self.pipeline.tokenizer is None and hasattr(self.pipeline, "tokenizer_2"):
            return True
        return False

    def is_valid_pipeline(self, pipeline):
        if not hasattr(pipeline, "tokenizer") and not hasattr(
            pipeline, "tokenizer_2"
        ):
            raise Exception(
                f"Cannot use PromptManipulation on a model without a tokenizer."
            )

    def process_long_prompt(self, positive_prompt: str, negative_prompt: str):
        batch_size = config.maximum_batch_size()
        if self.has_dual_text_encoders(self.pipeline):
            logging.debug(f'Running dual encoder Compel pipeline for batch size {batch_size}.')
            # We need to make a list of positive_prompt * batch_size count.
            positive_prompt = [positive_prompt] * batch_size
            conditioning, pooled_embed = self.compel(positive_prompt)
            negative_prompt = [negative_prompt] * batch_size
            negative_conditioning, negative_pooled_embed = self.compel(negative_prompt)
        else:
            logging.debug(f'Running single encoder Compel pipeline.')
            conditioning = self.compel.build_conditioning_tensor(positive_prompt)
            negative_conditioning = self.compel.build_conditioning_tensor(negative_prompt)
        [
            conditioning,
            negative_conditioning,
        ] = self.compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )
        if self.has_dual_text_encoders(self.pipeline):
            logging.debug(f'Returning pooled embeds along with positive/negative conditionings.')
            return conditioning, negative_conditioning, pooled_embed, negative_pooled_embed
        return conditioning, negative_conditioning

    @staticmethod
    def remove_duplicate_prompts(prompt: str, user_config: dict):
        to_remove = [
            user_config.get('positive_prompt', ''),
            user_config.get('negative_prompt', '')
        ]
        for segment in to_remove:
            if segment in prompt:
                prompt = prompt.replace(segment, '')
        return prompt
    
    @staticmethod
    def stylize_prompt(user_prompt: str, user_negative: str, user_style: str = None):
        logging.debug(f'Beginning stylize_prompt.')
        if user_style is not None:
            logging.debug(f'Received a user_style: {user_style}')
            if user_style not in prompt_styles:
                logging.error(f'Received invalid user style: {user_style}')
                raise ValueError(f'Invalid prompt style: {user_style}')
            if user_style == 'base':
                logging.warning(f'User is using base style. Returning prompts as-is.')
                return user_prompt, user_negative
        user_prompt, user_style = PromptManipulation.get_override_style(user_prompt, user_style)
        logging.info(f'Prompt style override: {user_style} for prompt: {user_prompt}')
        if user_style is not None:
            logging.debug(f'Prompt style override found: {user_style}')
            user_prompt = prompt_styles[user_style][0].format(prompt=user_prompt)
            user_negative = prompt_styles[user_style][1].format(prompt=user_negative)
        else:
            logging.warning(f'Not doing anything to the user prompt.')
        return user_prompt, user_negative
    
    @staticmethod
    def get_override_style(user_prompt: str, user_style: str):
        # A user can provide `--style [prompt]` in their user_prompt. We will replace this!
        if '--style' in user_prompt:
            user_style = user_prompt.split('--style')[1].split(' ')[1]
            user_prompt = user_prompt.replace(f'--style {user_style}', '')
        return user_prompt, user_style
    
# Path: discord_tron_client/classes/image_manipulation/diffusion.py