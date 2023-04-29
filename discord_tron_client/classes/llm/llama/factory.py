from discord_tron_client.classes.app_config import AppConfig

from discord_tron_client.classes.llm.llama.runner import LlamaRunner
from discord_tron_client.classes.llm.llama.cpp import LlamaCpp

config = AppConfig()
driver = config.llama_subsystem_type()
driver_mappings = {
    "llama.cpp": LlamaCpp
}

class LlamaFactory:
    def __init__(self):
        self.driver = driver_mappings[driver]()

    @staticmethod        
    def get_driver():
        return driver_mappings[driver]()
    
    @staticmethod
    def get() -> LlamaRunner:
        return LlamaRunner(LlamaFactory.get_driver())