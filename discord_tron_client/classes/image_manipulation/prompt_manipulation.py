from compel import Compel
from torch import Generator
import torch


# Manipulating prompts for the pipeline.
class PromptManipulation:
    def __init__(self, pipeline, device, use_second_encoder: bool = False):
        self.is_valid_pipeline(pipeline)
        self.pipeline = pipeline
        if (use_second_encoder and self.has_dual_text_encoders(pipeline)) or (
            self.has_dual_text_encoders(pipeline) and self.is_sdxl_refiner(pipeline)
        ):
            # SDXL Refiner and Base can both use the 2nd tokenizer/encoder.
            pipe_tokenizer = self.pipeline.tokenizer_2
            pipe_text_encoder = self.pipeline.text_encoder_2
        else:
            # Any other pipeline uses the first tokenizer/encoder.
            pipe_tokenizer = self.pipeline.tokenizer
            pipe_text_encoder = self.pipeline.text_encoder

        self.compel = Compel(
            tokenizer=pipe_tokenizer,
            text_encoder=pipe_text_encoder,
            truncate_long_prompts=False,
            device="cuda",
        )

    def has_dual_text_encoders(self, pipeline):
        return hasattr(pipeline, "text_encoder_2")

    def is_sdxl_refiner(self, pipeline):
        # SDXL Refiner has the 2nd text encoder, only.
        if self.pipeline.tokenizer is None and hasattr(self.pipeline, "tokenizer_2"):
            return True
        return False

    def is_valid_pipeline(self, pipeline):
        if not hasattr(self.pipeline, "tokenizer") and not hasattr(
            self.pipeline, "tokenizer_2"
        ):
            raise Exception(
                f"Cannot use PromptManipulation on a model without a tokenizer."
            )

    def process(self, prompt: str):
        conditioning = self.compel.build_conditioning_tensor(prompt, use_penultimate_clip_layer=True)
        return conditioning

    def process_long_prompt(self, positive_prompt: str, negative_prompt: str):
        if positive_prompt == "":
            positive_prompt = "since you did not provide a string, i will do it for you"
        conditioning = self.compel.build_conditioning_tensor(positive_prompt, use_penultimate_clip_layer=True)
        negative_conditioning = self.compel.build_conditioning_tensor(negative_prompt, use_penultimate_clip_layer=True)
        [
            conditioning,
            negative_conditioning,
        ] = self.compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )

        return conditioning, negative_conditioning


# Path: discord_tron_client/classes/image_manipulation/diffusion.py
