import logging, json, requests, sys, os, io, time
from discord_tron_client.classes.auth import Auth
from discord_tron_client.classes.app_config import AppConfig
from PIL import Image
import urllib3

class ApiClient:
    def __init__(self, auth: Auth, config: AppConfig):
        self.auth = auth
        self.config = config
        self.base_url = config.get_master_url()
        self.verify_ssl = config.verify_master_ssl()
        self.api_key = config.get_master_api_key()
        self.headers = self._set_auth_header()

    def update_auth(self):
        self._set_auth_header()

    def get(self, endpoint: str, params: dict = None):
        if params is None:
            params = {}
        url = self.base_url + endpoint
        params["api_key"] = self.api_key
        params["access_token"] = self.auth.get()
        response = requests.get(url, params=params, verify=self.verify_ssl)
        return self.handle_response(response)

    def put(self, endpoint: str, params: dict = None):
        if params is None:
            params = {}
        url = self.base_url + endpoint
        params["api_key"] = self.api_key
        params["access_token"] = self.auth.get()
        response = requests.put(url, params=params, verify=self.verify_ssl)
        return self.handle_response(response)

    def post(
        self,
        endpoint: str,
        params: dict = None,
        files: dict = None,
        send_auth: bool = True,
    ):
            try:
                logging.debug(f'Sending POST request to {endpoint} with files {files} and headers {self.headers} using params {params}')
                if params is None:
                    params = {}
                if send_auth:
                    self.headers = self._set_auth_header()
                url = self.base_url + endpoint
                response = requests.post(
                    url,
                    timeout=60,
                    params=params,
                    verify=self.verify_ssl,
                    files=files,
                    headers=self.headers,
                )
                return self.handle_response(response)
            except Exception as e:
                logging.error("Error in ApiClient.post: " + str(e))
                try:
                    if "Authentication required" in str(e):
                        logging.error(
                            "Error is authentication related. Refreshing auth."
                        )
                        self.update_auth()
                    raise e
                except Exception as e2:
                    logging.error(
                        "Error in ApiClient.post when checking error: " + str(e2)
                    )
                    raise e2


    def send_file(self, endpoint: str, file_path: str):
        with open(file_path, "rb") as f:
            response = self.post(endpoint, files={"file": f})
        return response

    async def send_audio(
        self, endpoint: str, buffer: io.BytesIO, send_auth: bool = True
    ):
        import asyncio

        loop = asyncio.get_event_loop()
        logging.debug(f"Uploading audio: {buffer}")
        response = await loop.run_in_executor(
            AppConfig.get_image_worker_thread(),  # Use a dedicated image processing thread worker.
            self.post,
            endpoint,
            None,
            {"audio_buffer": buffer},
            send_auth,
        )
        return response

    async def send_pil_image(self, endpoint: str, image: Image, send_auth: bool = True, image_metadata: dict = {}):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        attempt = 0
        while attempt < 15:
            try:
                buffer.seek(0)
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    AppConfig.get_image_worker_thread(),  # Use a dedicated image processing thread worker.
                    self.post,
                    endpoint,
                    image_metadata,
                    {"image": buffer},
                    send_auth,
                )
                return response
            except Exception as e:
                attempt += 1
                sleep_time = 2 ** attempt
                logging.error(
                    f"Error in ApiClient.post. Sleeping for {sleep_time} seconds."
                )
                time.sleep(sleep_time)
                if attempt >= 15:
                    raise Exception(f"Upload failed after 15 attempts")
    def send_buffer(self, endpoint: str, buffer: io.BytesIO):
        response = self.post(endpoint, files={"file": buffer})
        return response

    def handle_response(self, response):
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Error: {}".format(response.text))

    def _set_auth_header(self) -> dict:
        # We need the token from self.auth.get() to be set as the Authorization header using the Bearing token type
        current_ticket = self.auth.get()
        if (
            current_ticket is None
            or "access_token" not in current_ticket
            or current_ticket["access_token"] == ""
        ):
            raise Exception(f"No auth ticket found: {current_ticket}")

        self.headers = {"Authorization": f"Bearer {current_ticket['access_token']}"}
        return self.headers
