from discord_tron_client.classes.app_config import AppConfig

from discord_tron_client.classes.llm.stable_vicuna.runner import StableVicunaRunner
from discord_tron_client.classes.llm.stable_vicuna.torch import StableVicunaTorch

config = AppConfig()
driver = config.stablevicuna_subsystem_type()
driver_mappings = {"stablevicuna": StableVicunaTorch}


class StableVicunaFactory:
    def __init__(self):
        self.driver = driver_mappings[driver]()

    @staticmethod
    def get_driver():
        return driver_mappings[driver]()

    @staticmethod
    def get() -> StableVicunaRunner:
        return StableVicunaRunner(StableVicunaFactory.get_driver())
