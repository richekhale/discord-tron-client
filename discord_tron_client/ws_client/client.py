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
                ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                ssl_context.load_cert_chain(websocket_config['server_cert_path'], websocket_config['server_key_path'])
                ssl_context.load_verify_locations(websocket_config['server_cert_path'])
                # Set the correct SSL/TLS version (You can change PROTOCOL_TLS to the appropriate version if needed)
                ssl_context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
                verify_ssl = config.config.get("websocket_hub", {}).get("verify_ssl", False)
                if not verify_ssl:
                    ssl_context.check_hostname = verify_ssl  # Disable hostname verification
                    ssl_context.verify_mode = ssl.CERT_NONE  # Disable certificate verification

            # Add the access token to the header
            access_token = config.get_auth_ticket().get("access_token", None)
            headers = {
                "Authorization": f"Bearer {access_token}",
            }
            logging.info(f"Connecting to {hub_url}...")
            # Set the logging level for the websockets library only
            websocket_logger = logging.getLogger('websockets')
            websocket_logger.setLevel(logging.DEBUG) 
            async with websockets.connect(hub_url, ssl=ssl_context, extra_headers=headers, max_size=33554432, ping_timeout=10) as websocket:
                AppConfig.set_websocket(websocket)
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
                    asyncio.create_task(processor.process_command(payload=payload, websocket=websocket))
        except asyncio.exceptions.IncompleteReadError as e:
            logging.warning(f"IncompleteReadError: {e}")
            # ... handle the situation as needed
        except websockets.exceptions.ConnectionClosedError as e:
            logging.warning(f"ConnectionClosedError: {e}")
            # ... handle the situation as needed
        except Exception as e:
            logging.error(f"Unhandled exception in handler: {e}")

        except Exception as e:
            import traceback
            logging.error(f"Fatal Error: {e}, traceback: {traceback.format_exc()}")
            await asyncio.sleep(5)
