from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.message.discord import DiscordMessage
from discord_tron_client.classes.debug import clean_traceback
import logging, asyncio
config = AppConfig()

class StableMLRunner:
    def __init__(self, stableml_driver):
        self.driver = stableml_driver
        try:
            if config.is_stableml_enabled():
                self.driver.load_model()
        except Exception as e:
            logging.error(f"Could not load StableML driver: {e}")
        
    def predict(self, prompt, user_config):
        return self.driver.predict(prompt, user_config)
    def usage(self):
        driver_usage = self.driver.get_usage()
        if driver_usage is None:
            return None
        time_duration = -1
        if "time_duration" in driver_usage:
            time_duration = driver_usage["time_duration"]
        driver_details = self.driver.details() or "`Unknown LLM driver`"
        output_text = f"`{int(time_duration)} seconds`"
        if "total_token_count" in driver_usage:
            output_text = f"{output_text} using `{driver_usage['total_token_count']} tokens`"
        output_text = f"{output_text} via {driver_details}"
            
        return output_text

    async def predict_handler(self, payload, websocket):
        # We extract the features from the payload and pass them onto the actual generator
        user_config = payload["config"]
        prompt = payload["prompt"]
        logging.debug(f"StableMLRunner predict_handler received prompt {prompt}")
        discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="edit", message="Thinking!")
        websocket = AppConfig.get_websocket()
        await websocket.send(discord_msg.to_json())
        try:
            loop = asyncio.get_event_loop()
            loop_return = await loop.run_in_executor(
                AppConfig.get_image_worker_thread(),  # Use a dedicated image processing thread worker.
                self.predict,
                prompt,
                user_config
            )
            logging.debug(f"StableMLRunner predict_handler received result {loop_return}")
            discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="send_large_message", message=f'<@{payload["discord_context"]["author"]["id"]}>: ' + '`' + prompt + '`\n' + loop_return)
            websocket = AppConfig.get_websocket()
            await websocket.send(discord_msg.to_json())
            usage = self.usage()
            if usage is not None:
                discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="send", message=f'<@{payload["discord_context"]["author"]["id"]}>: ' + 'Your prompt, `' + prompt + '`, used ' + f'{usage}')
                websocket = AppConfig.get_websocket()
                await websocket.send(discord_msg.to_json())

            discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="delete")
            websocket = AppConfig.get_websocket()
            await websocket.send(discord_msg.to_json())
            discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_context"], module_command="delete")
            websocket = AppConfig.get_websocket()
            await websocket.send(discord_msg.to_json())

        except Exception as e:
            import traceback
            logging.error(f"Received an error in StableMLRunner.predict_handler: {e}, traceback: {clean_traceback(traceback.format_exc())}")
            discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="edit", message="We pooped the bed!")
            websocket = AppConfig.get_websocket()
            await websocket.send(discord_msg.to_json())
            raise e