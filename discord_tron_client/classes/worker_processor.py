from discord_tron_client.classes.message import WebsocketMessage
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.message.job_queue import JobQueueMessage
from discord_tron_client.modules.image_generation import generator as image_generator
from discord_tron_client.modules.image_generation import variation
from discord_tron_client.classes.llm.llama.factory import LlamaFactory
from discord_tron_client.classes.llm.stablelm.factory import StableLMFactory
from discord_tron_client.classes.llm.stable_vicuna.factory import StableVicunaFactory
from discord_tron_client.classes.tts.bark.factory import BarkFactory
from typing import Dict, Any
import logging, json, websocket
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()
try:
    llamarunner = LlamaFactory.get()
except:
    logging.warn('Could not retrieve a Llama driver.')

try:
    stablelmrunner = StableLMFactory.get()
except:
    logging.warn('Could not retrieve a StableLM driver.')

try:
    stablevicunarunner = StableVicunaFactory.get()
except:
    logging.warn('Could not retrieve a StableVicuna driver.')

identifier = HardwareInfo.get_identifier()

class WorkerProcessor:
    def __init__(self):
        self.command_handlers = {
            "image_generation": {
                "generate_image": image_generator.generate_image,
            },
            "image_upscaling": {
                "upscale": image_generator.generate_image,
            },
            "image_variation": {
                "promptless_variation": variation.promptless_variation,
                "prompt_variation": variation.prompt_variation,
            },
            "llama":{
                "predict": llamarunner.predict_handler
            },
            "stablelm": {
                "predict": stablelmrunner.predict_handler
            },
            "stablevicuna": {
                "predict": stablevicunarunner.predict_handler
            },
            "tts_bark": {
                "generate": BarkFactory.get().generate_handler
            }
            # Add more command handlers as needed
        }

    async def process_command(self, payload: Dict, websocket: websocket) -> None:
        try:
            logging.debug(f"Entered process_command via WebSocket, payload: {payload}")
            if "module_name" not in payload:
                logging.warn("Not executing command payload via WorkerProcessor, as it does not contain a module_name: " + str(payload))
                return
            handler = self.command_handlers.get(payload["module_name"], {}).get(payload["module_command"])
            if handler is None:
                # No handler found for the command
                logging.error(f"No handler found in module " + str(payload["module_name"]) + " for command " + payload["module_command"] + ", payload: " + str(payload))
                return
            logging.info(f"Running handler for command: {payload['module_command']} in module: {payload['module_name']}")
            logging.debug("Executing incoming " + str(handler) + " for module " + str(payload["module_name"]) + ", command " + payload["module_command"] + ", payload: " + str(payload))
            handler_result = await handler(payload, websocket)

        except Exception as e:
            # enable tracemalloc:
            import tracemalloc
            tracemalloc.start()
            # take a snapshot:
            snapshot = tracemalloc.take_snapshot()
            # show top 10 lines
            top_stats = snapshot.statistics('lineno')
            for stat in top_stats[:10]:
                print(stat)
            # disable tracemalloc:
            tracemalloc.stop()
            import traceback
            logging.error(f"Error processing command: {e}, traceback: {traceback.format_exc()} ")
            
            return json.dumps({"error": str(e)})

        finally:
            if "job_id" in payload and payload["job_id"] != "":
                # We have the output, but now we need to mark the Job as finished
                discord_msg = JobQueueMessage(websocket=websocket, job_id=payload["job_id"], worker_id=identifier, module_command="finish")
                websocket = AppConfig.get_websocket()
                await websocket.send(discord_msg.to_json())
    # Add more command handler methods as needed
