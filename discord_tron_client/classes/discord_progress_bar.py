from typing import Dict
from discord_tron_client.message.discord import DiscordMessage
from discord_tron_client.classes.app_config import AppConfig
import logging, websockets, time, asyncio
from websockets.client import WebSocketClientProtocol


class DiscordProgressBar:
    def __init__(
        self,
        websocket: WebSocketClientProtocol,
        websocket_message: DiscordMessage,
        discord_first_message: Dict,
        progress_bar_steps=100,
        progress_bar_length=20,
    ):
        self.total_steps = progress_bar_steps
        self.progress_bar_length = progress_bar_length
        self.current_step = 0
        self.current_stage = 1
        self.current_stage_msg = ""
        self.websocket_msg = websocket_message
        self.websocket = websocket
        self.discord_first_message = discord_first_message
        # Last updated time.
        self.last_update = time.time()

    async def update_progress_bar(self, step: int):
        if step < self.current_step:
            logging.warn(
                f"Step {step} is less than current step {self.current_step}. This means the progress bar tried updating to the same state more than once."
            )
        self.current_step = step
        if step == 0 and self.current_step == 100:
            # We might be onto stage two of a multi-stage operation..
            self.current_stage += 1
            self.current_stage_msg = " (Stage " + str(self.current_stage) + ")"
            return
        progress = self.current_step / self.total_steps
        filled_length = int(progress * self.progress_bar_length)
        bar = "█" * filled_length + "-" * (self.progress_bar_length - filled_length)
        percent = round(progress * 100, 1)
        progress_text = "`" + f"[{bar}] {percent}% complete`"
        we_have_another_fifth_of_progress = percent % 30
        if we_have_another_fifth_of_progress == 0:
            try:
                # Update the websocket message template
                self.websocket_msg.update(
                    arguments={"message": progress_text + self.current_stage_msg}
                )
                to_send = self.websocket_msg.to_json()
                self.websocket = AppConfig.get_websocket()
                await self.send_update(
                    self.websocket, str(to_send)
                )  # Use the send_update function here
            except Exception as e:
                logging.error("Traceback: ", exc_info=True)

    async def send_update(self, websocket, message, max_retries=5):
        try:
            await websocket.send(message)
            await websocket.ping()
        except Exception as e:
            logging.error(f"Error sending message to websocket: {e}")

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
                "progress_bar_length": self.progress_bar_length,
            },
        }
