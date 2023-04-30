from discord_tron_client.classes.app_config import AppConfig
from bark.api import generate_audio
from bark.generation import preload_models
from bark.generation import SAMPLE_RATE
import os, sys, json, logging, time

config = AppConfig()
sample_text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""

class BarkTorch:
    def __init__(self):
        self.loaded = False
        self.model = 'Bark'

    def details(self):
        return f'PyTorch running the {self.model} audio generation model'

    def get_usage(self):
        return self.usage or None

    def load_model(self):
        if self.loaded:
            logging.debug(f"Not reloading Bark TTS models.")
            return
        logging.info(f"Loading Bark TTS model, as it was not already found loaded.")
        preload_models()
        self.loaded = True

    def _generate(self, prompt, user_config):
        # generate audio from text
        audio = generate_audio(prompt)
        return audio

    def generate(self, prompt, user_config):
        logging.debug(f"Begin Bark generate() routine")
        time_begin = time.time()
        # User settings overrides.
        audio, semantic_x = self._generate(prompt=prompt, user_config=user_config)
        time_end = time.time()
        time_duration = time_end - time_begin
        logging.debug(f"Completed generation in {time_duration} seconds: {audio}")
        if audio is None:
            raise RuntimeError(f"{self.model} returned no result.")
        self.usage = {"time_duration": time_duration}
        return audio, SAMPLE_RATE