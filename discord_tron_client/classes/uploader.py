from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image as PILImage
from discord_tron_client.classes.auth import Auth
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.api_client import ApiClient
from typing import List
from io import BytesIO
import logging, json, asyncio, base64, urllib3
from scipy.io.wavfile import write as write_wav

urllib3.disable_warnings()
config = AppConfig()

semaphore = asyncio.Semaphore(config.get_max_concurrent_uploads())


class Uploader:
    def __init__(self, api_client: ApiClient, config: AppConfig):
        self.api_client = api_client
        self.config = config

    def start_thread_pool(self, num_workers=4):
        self.thread_pool = ThreadPool(processes=num_workers)
        return self.thread_pool

    def run_threads(self):
        self.thread_pool.close()
        self.thread_pool.join()

    def image(self, image):
        logging.debug(f"Uploading image to {self.config.get_master_url()}")
        self.api_client.update_auth()
        result = asyncio.run(
            self.api_client.send_pil_image(
                "/upload_image",
                image,
                False,
                getattr(image, "info", {"error": "no_metadata"}),
            )
        )
        logging.debug(f"Image uploader received result: {result}")
        if "image_url" in result:
            return result["image_url"]
        raise Exception(f"Image upload failed: {result}")

    def video(self, video_path: str):
        logging.debug(
            f"Uploading video from path {video_path} to {self.config.get_master_url()}"
        )
        self.api_client.update_auth()
        result = asyncio.run(self.api_client.send_file("/upload_video", video_path))
        logging.debug(f"Received response from upload video endpoint:  {result}")
        return result.get("video_url")

    async def upload_images(self, images: List):
        self.start_thread_pool(len(images))
        results = self.thread_pool.map(self.image, images)
        self.run_threads()
        return results

    async def upload_videos(self, video_path: str):
        images = [video_path]
        self.start_thread_pool(len(images))
        results = self.thread_pool.map(self.video, images)
        self.run_threads()
        return results

    async def audio(self, audio_data, sample_rate):
        logging.debug(f"Uploading audio to {self.config.get_master_url()}")
        self.api_client.update_auth()
        wav_binary_stream = BytesIO()
        write_wav(wav_binary_stream, sample_rate, audio_data)
        # Reset the binary stream's position to the beginning
        wav_binary_stream.seek(0)
        result = await self.api_client.send_audio(
            "/upload_audio", wav_binary_stream, False
        )
        logging.debug(f"Audio uploader received result: {result}")
        if "audio_url" in result:
            return result["audio_url"]
        raise Exception(f"Audio upload failed: {result}")

    async def upload_audio_files(self, audio_files_data: List, sample_rates: List):
        self.start_thread_pool(len(audio_files_data))
        results = self.thread_pool.map(self.audio, audio_files_data, sample_rates)
        self.run_threads()
        return results
