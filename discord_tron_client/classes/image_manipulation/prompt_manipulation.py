from compel import Compel
from torch import Generator
import torch


# Manipulating prompts for the pipeline.
class PromptManipulation:
    def __init__(self, pipeline, device):
        self.pipeline = pipeline
        if not hasattr(self.pipeline, "tokenizer") and not hasattr(self.pieline, "tokenizer_2"):
            raise Exception(
                f"Cannot use PromptManipulation on a model without a tokenizer."
            )
        pipe_tokenizer = self.pipeline.tokenizer
        if self.pipeline.tokenizer is None and hasattr(self.pipeline, 'tokenizer_2'):
            pipe_tokenizer = self.pipeline.tokenizer_2
        self.compel = Compel(
            tokenizer=pipe_tokenizer,
            text_encoder=pipeline.text_encoder,
            truncate_long_prompts=False,
            device="cuda"
        )

    def process(self, prompt: str):
        conditioning = self.compel.build_conditioning_tensor(prompt)
        return conditioning

    def process_long_prompt(self, positive_prompt: str, negative_prompt: str):
        if positive_prompt == "":
            positive_prompt = "since you did not provide a string, i will do it for you"
        conditioning = self.compel.build_conditioning_tensor(positive_prompt)
        negative_conditioning = self.compel.build_conditioning_tensor(negative_prompt)
        [
            conditioning,
            negative_conditioning,
        ] = self.compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )

        return conditioning, negative_conditioning


# Path: discord_tron_client/classes/image_manipulation/diffusion.py
