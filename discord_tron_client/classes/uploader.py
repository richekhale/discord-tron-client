# Upload files to the central API.
from PIL import Image
from discord_tron_client.classes.auth import Auth
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.api_client import ApiClient
import logging, json

class Uploader:
    def __init__(self, api_client: ApiClient, config: AppConfig):
        self.api_client = api_client
        self.config = config
        
    async def image(self, image: Image):
        logging.debug(f"Uploading image to {self.config.get_master_url()}")
        result = await self.api_client.send_pil_image('/upload_image', image)
        logging.debug(f"Image uploader received result: {result}")
        
        if "image_url" in result:
            return result["image_url"]
        raise Exception(f"Image upload failed: {result}")