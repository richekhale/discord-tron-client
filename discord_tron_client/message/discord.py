from discord_tron_client.classes.message import WebsocketMessage
from typing import Dict
import logging

class DiscordMessage(WebsocketMessage):
    def __init__(self, message: str, context: Dict, module_command: str = "send"):
        message_type="discord"
        module_name="message"
        super().__init__(message_type, module_name, module_command, data=context, arguments={"message": message})
        
    async def send(self, websocket):
        logging.info(f"Sending request for Discord: {self.to_json()}")
        await websocket.send(self.to_json())