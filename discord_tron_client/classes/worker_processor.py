from discord_tron_client.classes.message import WebsocketMessage
from discord_tron_client.modules.image_generation import generator as image_generator
from typing import Dict, Any
import logging, json, websocket

class WorkerProcessor:
    def __init__(self):
        self.command_handlers = {
            "image_generation": {
                "generate_image": image_generator.generate_image
            },
            # Add more command handlers as needed
        }

    async def process_command(self, payload: Dict, websocket: websocket) -> None:
        try:
            logging.info(f"Entered process_command via WebSocket, payload: {payload}")
            if "module_name" not in payload:
                logging.warn("Not executing command payload via WorkerProcessor, as it does not contain a module_name: " + str(payload))
                return
            handler = self.command_handlers.get(payload["module_name"], {}).get(payload["module_command"])
            if handler is None:
                # No handler found for the command
                logging.error(f"No handler found in module " + str(payload["module_name"]) + " for command " + payload["module_command"] + ", payload: " + str(payload))
                return
            logging.info("Executing incoming " + str(handler) + " for module " + str(payload["module_name"]) + ", command " + payload["module_command"] + ", payload: " + str(payload))
            return await handler(payload, websocket)
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
            
            logging.error("Error processing command: " + str(e))
            
            return json.dumps({"error": str(e)})
    # Add more command handler methods as needed
