from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image as PILImage
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
        
    def image(self, image):
        logging.debug(f"Uploading image to {self.config.get_master_url()}")
        self.api_client.update_auth()
        result = asyncio.run(self.api_client.send_pil_image('/upload_image', image, False))
        logging.debug(f"Image uploader received result: {result}")
        if "image_url" in result:
            return result["image_url"]
        raise Exception(f"Image upload failed: {result}")

    def start_thread_pool(self, num_workers=4):
        self.thread_pool = ThreadPool(processes=num_workers)
        return self.thread_pool
    def run_threads(self):
        self.thread_pool.close()
        self.thread_pool.join()

    async def upload_images(self, images: List):
        self.start_thread_pool(len(images))
        results = self.thread_pool.map(self.image, images)
        self.run_threads()
        return results
