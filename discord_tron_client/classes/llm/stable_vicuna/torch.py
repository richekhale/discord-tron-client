from discord_tron_client.classes.app_config import AppConfig
import discord_tron_client.classes.llm.stable_vicuna.helpers.pipeline as predict

import os, sys, json, logging, time

config = AppConfig()

class StableVicunaTorch:
    def __init__(self):
        self.model = config.stablevicuna_model_default()
        self.model_config = None
        self.tokenizer = None
        self.vicuna = None

    def load_model(self):
        if not config.is_stablevicuna_enabled():
            return
        self.tokenizer, self.vicuna, self.max_context_length = predict.load(self.model)

    def details(self):
        return f'StableVicuna running the {self.model} parameter model via 🤗 Diffusers'

    def get_usage(self):
        return self.usage or None

    def _predict(self, user_config, prompt, history = None, seed = 1337, max_tokens = 512, temperature = 0.8, repeat_penalty = 1.1, top_p = 0.95, top_k=40):
        return predict.generate(tokenizer=self.tokenizer, model=self.vicuna, user_config=user_config, user_prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, seed=seed)

    def predict(self, prompt, history, user_config, max_tokens = 4096, temperature = 1.0, repeat_penalty = 1.1, top_p = 0.95, top_k=40):
        logging.debug(f"Begin StableVicuna prediction routine")
        logging.debug(f"Our received parameters: max_tokens {max_tokens} top_p {top_p} top_k {top_k} repeat_penalty {repeat_penalty} temperature {temperature}")
        time_begin = time.time()
        # User settings overrides.
        seed = user_config.get("seed", None)
        temperature = user_config.get("temperature", temperature)
        top_p = user_config.get("top_p", top_p)
        top_k = user_config.get("top_k", top_k)
        repeat_penalty = user_config.get("repeat_penalty", repeat_penalty)
        # Maybe the user wants fewer tokens.
        user_max_tokens = user_config.get("max_tokens", max_tokens)
        if max_tokens >= user_max_tokens:
            max_tokens = user_max_tokens
        logging.debug(f"Our post-override parameters: max_tokens {max_tokens} top_p {top_p} top_k {top_k} repeat_penalty {repeat_penalty} temperature {temperature}")
        logging.debug("Beginning StableVicuna prediction..")
        llm_result, token_count = self._predict(prompt=prompt, history=history, user_config=user_config, seed=seed, max_tokens=max_tokens, temperature=temperature, repeat_penalty=repeat_penalty, top_p=top_p, top_k=top_k)
        time_end = time.time()
        time_duration = time_end - time_begin
        logging.debug(f"Completed prediction in {time_duration} seconds: {llm_result}")
        if llm_result is None:
            raise RuntimeError("StableVicuna returned no result.")
        self.usage = {"time_duration": time_duration, "total_token_count": token_count}

        return llm_result
    
    def set_seed(self):
        # If the user has not specified a seed, we will use the current time.
        if seed is None or seed == 0:
            logging.debug("Timestamp being used as the seed.")
            seed = int(time_begin)
        elif seed == -1:
            # -1 is a special condition for randomizing the seed more than just using the timestamp.
            logging.debug("Ultra-seed randomizer engaged!")
            import random
            seed = random.randint(0, 999999999)
        else:
            logging.debug("A pre-selected seed was provided.")
        logging.debug(f"Seed chosen: {seed}")
