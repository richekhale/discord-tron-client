from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.tts.bark.runner import BarkRunner
from discord_tron_client.classes.tts.bark.torch import BarkTorch

config = AppConfig()
driver = config.bark_subsystem_type()
driver_mappings = {
    "torch": BarkTorch
}

class BarkFactory:
    def __init__(self):
        self.driver = driver_mappings[driver]()

    @staticmethod        
    def get_driver():
        return driver_mappings[driver]()
    
    @staticmethod
    def get() -> BarkRunner:
        return BarkRunner(BarkFactory.get_driver())