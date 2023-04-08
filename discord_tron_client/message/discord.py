from discord_tron_client.classes.message import WebsocketMessage
from typing import Dict
from PIL import Image
import logging, websocket
import base64
from io import BytesIO

class DiscordMessage(WebsocketMessage):
    def __init__(self, websocket: websocket,  context, module_command: str = "send", message: str = "Loading the model and preparing to generate your image!", image: Image = None):
        self.websocket = websocket
        if isinstance(context, DiscordMessage):
            # Extract the context from the existing DiscordMessage
            context = context.data
            logging.info(f"Extracted data from the DiscordMessage context: {context}")
        arguments = { "message": message }
        if image is not None:
            arguments["image"] = self.b64_image(image)
        super().__init__(message_type="discord", module_name="message", module_command=module_command, data=context, arguments=arguments)
        
    def b64_image(self, image: Image):
        # Save image to buffer before encoding as base64:

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return b64_image