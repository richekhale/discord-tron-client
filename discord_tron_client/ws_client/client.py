import json, logging
logging.basicConfig(level=logging.INFO)
import ssl, websockets, asyncio
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.message import WebsocketMessage
from discord_tron_client.classes.worker_processor import WorkerProcessor

async def websocket_client(config: AppConfig, startup_sequence:str = None):
    processor = WorkerProcessor()
    while True:
        try:
            websocket_config = config.get_websocket_config()
            logging.debug(f"Retrieved websocket config: {websocket_config}")
            hub_url = str(websocket_config["protocol"]) + "://" + str(websocket_config["host"]) + ":" + str(websocket_config["port"])
            tls = websocket_config['tls']
            ssl_context = None
            if tls:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_verify_locations(websocket_config['server_cert_path'])

            # Add the access token to the header
            access_token = config.get_auth_ticket().get("access_token", None)
            headers = {
                "Authorization": f"Bearer {access_token}",
            }
            logging.info(f"Connecting to {hub_url}...")
            async with websockets.connect(hub_url, ssl=ssl_context, extra_headers=headers, max_size=33554432) as websocket:
                # Send the startup sequence
                if startup_sequence:
                    for message in startup_sequence:
                        logging.debug(f"Sending startup sequence message: {message}")
                        await websocket.send(message.to_json())
                    if message:
                        del message
                else:
                    logging.error("No startup sequence found.")
                async for message in websocket:
                    logging.info(f"Received message from master")
                    logging.debug(f"{message}")
                    payload = json.loads(message)
                    await processor.process_command(payload=payload, websocket=websocket)
        except Exception as e:
            import traceback
            logging.error(f"Fatal Error: {e}, traceback: {traceback.format_exc()}")
            await asyncio.sleep(5)
