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

    def _generate(self, prompt, user_config, character_voice: str = None):
        # generate audio from text
        if character_voice is None:
            logging.debug(f"No voice provided, using default.")
            character_voice = user_config.get("tts_voice", "en_female_intense")
        logging.debug(f"Generating text {prompt[32:]}.. with voice {character_voice}")

        audio, semantics = generate_audio(prompt, confused_travolta_mode=user_config.get("confused_tts", False), history_prompt=character_voice, text_temp=user_config.get("temperature", 0.7), waveform_temp=user_config.get("waveform_temp", 0.7))
        return audio, semantics

    def generate(self, prompt, user_config):
        logging.debug(f"Begin Bark generate() routine")
        time_begin = time.time()
        # User settings overrides.
        audio, _, semantic_x = self.generate_long(prompt=prompt, user_config=user_config)
        time_end = time.time()
        time_duration = time_end - time_begin
        logging.debug(f"Completed generation in {time_duration} seconds: {audio}")
        if audio is None:
            raise RuntimeError(f"{self.model} returned no result.")
        self.usage = {"time_duration": time_duration}

        return audio, SAMPLE_RATE, semantic_x
    
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
        return self.generate_long_from_segments(segments, user_config)       

    def generate_long_from_segments(self, prompts: List[str], user_config):
        # Generate audio for each prompt
        audio_segments = []
        actors = user_config.get("tts_actors", None)
        logging.debug(f"Generating long prompt with {len(prompts)} segments. using actors {actors}")
        current_voice = None
        for prompt in prompts:
            line, voice = BarkTorch.process_line(prompt, actors)
            if voice is not None:
                # Set a voice, if found. Otherwise, keep last voice.
                current_voice = voice
            audio, _, semantics = self._generate(line, user_config, current_voice)
            audio_segments.append(audio)
        # Concatenate the audio segments
        concatenated_audio = self.concatenate_audio_segments(audio_segments)
        return concatenated_audio, SAMPLE_RATE, semantics

    @staticmethod
    def concatenate_audio_segments(audio_segments):
        combined_audio = np.array([], dtype=np.int16)

        for audio in audio_segments:
            # Concatenate the audio
            combined_audio = np.concatenate((combined_audio, audio))

        return combined_audio

    @staticmethod
    def process_line(line, characters):
        if characters is None:
            return line, None
        pattern = r"\{([^}]+)\}:?"
        match = re.search(pattern, line)
        if match:
            actor = match.group(1)
            line = re.sub(pattern, "", line).strip()
            # This can strip out "not-found" {STRINGS} so beware...
            character_voice = characters.get(actor, {}).get("voice", None)
        else:
            character_voice = None
        return line, character_voice