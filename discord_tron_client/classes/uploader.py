from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from discord_tron_client.classes.auth import Auth
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.api_client import ApiClient
from typing import List
import logging, json, asyncio
config = AppConfig()

semaphore = asyncio.Semaphore(config.get_max_concurrent_uploads())

class Uploader:
    def __init__(self, api_client: ApiClient, config: AppConfig):
        self.api_client = api_client
        self.config = config
        
    async def image(self, image: Image):
        logging.debug(f"Uploading image to {self.config.get_master_url()}")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.thread_pool, self.api_client.send_pil_image, '/upload_image', image)
        logging.debug(f"Image uploader received result: {result}")
        
        if "image_url" in result:
            return result["image_url"]
        raise Exception(f"Image upload failed: {result}")

    def start_thread_pool(self, num_workers=4):
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        return self.thread_pool

    async def upload_images(self, images: List):
        self.start_thread_pool(len(images))
        tasks = [await self.image(img) for img in images]
        results = await asyncio.gather(*tasks)
        return results
