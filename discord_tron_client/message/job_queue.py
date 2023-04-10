from discord_tron_client.classes.message import WebsocketMessage
from typing import Dict
from PIL import Image
import logging, websocket
import base64
from io import BytesIO
from discord_tron_client.classes.hardware import HardwareInfo
hardware = HardwareInfo()

class JobQueueMessage(WebsocketMessage):
    def __init__(self, websocket: websocket,  job_id: str, worker_id: str, module_command: str):
        self.websocket = websocket
        arguments = { "job_id": job_id, "worker_id": worker_id }
        super().__init__(message_type="job", module_name="job_queue", module_command=module_command, data={}, arguments=arguments)
