from discord_tron_client.classes.app_config import AppConfig

from discord_tron_client.classes.stableml.runner import StableMLRunner
from discord_tron_client.classes.stableml.py import StableMLPy

config = AppConfig()
driver = config.stableml_subsystem_type()
driver_mappings = {
    "stableml.py": StableMLPy
}

class StableMLFactory:
    def __init__(self):
        self.driver = driver_mappings[driver]()

    @staticmethod        
    def get_driver():
        return driver_mappings[driver]()
    
    @staticmethod
    def get() -> StableMLRunner:
        return StableMLRunner(StableMLFactory.get_driver())