from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.llm.llama.code import helper

import os, sys, json, logging, time

config = AppConfig()


class LlamaCode:
    def __init__(self):
        self.model = helper.llm_model_name

    def details(self):
        return f"PyTorch running the {self.model} parameter model"

    def get_usage(self):
        return self.usage or None

    def predict(
        self,
        prompt,
        user_config,
        max_tokens=4096,
        temperature=1.0,
        repeat_penalty=1.1,
        top_p=0.95,
        top_k=40,
    ):
        logging.debug(f"Begin LlamaCpp prediction routine")

        logging.debug(
            f"Our received parameters: max_tokens {max_tokens} top_p {top_p} top_k {top_k} repeat_penalty {repeat_penalty} temperature {temperature}"
        )
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
        logging.debug(
            f"Our post-override parameters: max_tokens {max_tokens} top_p {top_p} top_k {top_k} repeat_penalty {repeat_penalty} temperature {temperature}"
        )

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

        logging.debug("Beginning Llama.cpp prediction..")
        cpp_result = self._predict(
            prompt=prompt,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            top_p=top_p,
            top_k=top_k,
        )
        time_end = time.time()
        time_duration = time_end - time_begin
        logging.debug(f"Completed prediction in {time_duration} seconds: {cpp_result}")
        if cpp_result is None:
            raise RuntimeError("LLaMA.cpp returned no result.")
        if "choices" not in cpp_result:
            raise RuntimeError("LLaMA.cpp returned an invalid result.")
        if (
            "text" not in cpp_result["choices"][0]
            or cpp_result["choices"][0]["text"] == ""
        ):
            raise RuntimeError("LLaMA.cpp returned an empty set.")
        if "finish_reason" not in cpp_result["choices"][0]:
            logging.warn(f"LLaMA.cpp did not return a finish_reason: {cpp_result}")
        self.usage = {"time_duration": time_duration}
        if "usage" in cpp_result:
            self.usage.update(cpp_result["usage"])
        return cpp_result["choices"][0]["text"]
