from typing import Dict
from discord_tron_client.message.discord import DiscordMessage
from discord_tron_client.classes.app_config import AppConfig
import logging, websockets, time
from websockets.client import WebSocketClientProtocol
class DiscordProgressBar:
    def __init__(self, websocket: WebSocketClientProtocol, websocket_message: DiscordMessage, discord_first_message: Dict, progress_bar_steps = 100, progress_bar_length = 20):
        self.total_steps = progress_bar_steps
        self.progress_bar_length = progress_bar_length
        self.current_step = 0
        self.websocket_msg = websocket_message
        self.websocket = websocket
        self.discord_first_message = discord_first_message
        # Last updated time.
        self.last_update = 0
    async def update_progress_bar(self, step: int):
        if step < self.current_step:
            # We do not want time going backwards for a progress bar.
            logging.debug("Time went backwards, Marty!")
            return
        self.current_step = step
        progress = self.current_step / self.total_steps
        filled_length = int(progress * self.progress_bar_length)
        bar = "█" * filled_length + "-" * (self.progress_bar_length - filled_length)
        percent = round(progress * 100, 1)
        progress_text = "`" + f"[{bar}] {percent}% complete`"
        we_have_another_fifth_of_progress = percent % 20
        # Let's not accidentally trigger too many updates. Store the time here, and wait at least 5 seconds before another update.
        current_Time = time.time()
        if current_Time - self.last_update < 5:
            return
        # We have passed five seconds. Update can continue. Mark new time.
        self.last_update = current_Time
        if we_have_another_fifth_of_progress == 0:
            logging.debug(f"Current document for websocket_msg: {self.websocket_msg.to_json()}")
            logging.debug(f"Update variables: {progress}, {filled_length}, {bar}, {percent}, {progress_text}, {we_have_another_fifth_of_progress}")
            # await self.discord_first_message.edit(content=progress_text)
            logging.info("Sending progress bar to websocket!")
            try:
                # Update the websocket message template
                self.websocket_msg.update(arguments={"message": progress_text})
                to_send = self.websocket_msg.to_json()
                logging.debug(f"Sending data: {to_send}")
                try:
                    await self.websocket.send(str(to_send))
                except websockets.exceptions.ConnectionClosedError as e:
                    logging.error("Connection closed while sending progress bar update! Retrieving fresh websockie?")
                    self.websocket = AppConfig.get_websocket()
                    logging.error("Traceback: ", exc_info=True)
            except Exception as e:
                logging.error("Traceback: ", exc_info=True)
    
    # Return a JSON representation of the object
    def to_json(self):
        return {
            "type": "discord",
            "module": "progress_bar",
            "command": "update",
            "data": {
                "discord_first_message": self.discord_first_message,
                "progress": self.current_step,
                "total_steps": self.total_steps,
                "progress_bar_length": self.progress_bar_length
            }
        }