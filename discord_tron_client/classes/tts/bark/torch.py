from discord_tron_client.classes.app_config import AppConfig
from bark.api import generate_audio
from bark.generation import preload_models
from bark.generation import SAMPLE_RATE
import os, sys, json, logging, time, io, re
from pydub import AudioSegment
from scipy.io.wavfile import write as write_wav
from typing import List
import numpy as np

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
        longer_than_14_seconds, estimated_time = self.estimate_spoken_time(prompt)
        print(f"Estimated time: {estimated_time:.2f} seconds.")
        if longer_than_14_seconds:
            print(f"Text Prompt could be too long, might want to try a shorter one if you get a bad result.")
            print("now split the text_prompt to less than 14 seconds asap")
            text_prompts_to_process, Len = self.split_text_prompt(prompt)
            print(f"split text_prompt to {Len} segments")

        print(f"Generating: {prompt}")

        audio = generate_audio(prompt, confused_travolta_mode=user_config.get("confused_tts", False), history_prompt=user_config.get("persona_tts", "en_fiery"), text_temp=user_config.get("temperature", 0.7), waveform_temp=user_config.get("waveform_temp", 0.7))
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
    
    def split_text_prompt(self, text_prompt, maxword=30):
        text_prompt = re.sub(r'\s{2,}', ' ', text_prompt)
        segments = re.split(r'(?<=[,.])\s*', text_prompt)
        segments = [re.sub(r'[^a-zA-Z0-9,. ]', '', segment) for segment in segments]

        result = []
        buffer = ""
        for segment in segments:
            words = segment.split()

            if len(buffer.split()) + len(words) > maxword:
                while len(words) > maxword:
                    result.append(' '.join(words[:maxword]) + '.')
                    words = words[maxword:]
            if len(buffer.split()) + len(words) < 15:
                buffer += " " + segment
            else:
                result.append(buffer.strip() + segment)
                buffer = ""

        if buffer:
            result.append(buffer.strip())

        result = [segment.rstrip(',') + '.' if not segment.endswith('.') else segment for segment in result]

        return result, len(result)
    def estimate_spoken_time(self, text, wpm=150, time_limit=14):
        # Remove text within square brackets
        text_without_brackets = re.sub(r'\[.*?\]', '', text)
        
        words = text_without_brackets.split()
        word_count = len(words)
        time_in_seconds = (word_count / wpm) * 60
        
        if time_in_seconds > time_limit:
            return True, time_in_seconds
        else:
            return False, time_in_seconds
    def generate_long(self, prompt, user_config):
        # Split the prompt into smaller segments
        segments, _ = self.split_text_prompt(prompt)
        
        # Generate audio for each segment
        audio_segments = []
        for segment in segments:
            audio, _ = self.generate(segment, user_config)
            audio_segments.append(audio)

        # Concatenate the audio segments
        concatenated_audio = self.concatenate_audio_segments(audio_segments)

        return concatenated_audio, SAMPLE_RATE

    def generate_long_from_segments(self, prompts: List[str], user_config):
        # Generate audio for each prompt
        audio_segments = []
        for prompt in prompts:
            audio, _ = self.generate(prompt, user_config)
            audio_segments.append(audio)

        # Concatenate the audio segments
        concatenated_audio = self.concatenate_audio_segments(audio_segments)

        return concatenated_audio, SAMPLE_RATE
    @staticmethod
    def concatenate_audio_segments(audio_segments):
        combined_audio = np.array([], dtype=np.int16)

        for audio in audio_segments:
            # Concatenate the audio
            combined_audio = np.concatenate((combined_audio, audio))

        return combined_audio