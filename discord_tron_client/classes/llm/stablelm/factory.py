from discord_tron_client.classes.app_config import AppConfig

from discord_tron_client.classes.llm.stablelm.runner import StableLMRunner
from discord_tron_client.classes.llm.stablelm.py import StableLMPy

config = AppConfig()
driver = config.stablelm_subsystem_type()
driver_mappings = {"stablelm.py": StableLMPy}


class StableLMFactory:
    def __init__(self):
        self.driver = driver_mappings[driver]()

    @staticmethod
    def get_driver():
        return driver_mappings[driver]()

    @staticmethod
    def get() -> StableLMRunner:
        return StableLMRunner(StableLMFactory.get_driver())
