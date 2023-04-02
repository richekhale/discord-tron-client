import json, logging
import ssl, websockets
from discord_tron_client.classes.app_config import AppConfig

async def websocket_client(config: AppConfig, startup_sequence:str = None):
    websocket_config = config.get_websocket_config()
    logging.info(f"Retrieved websocket config: {websocket_config}")
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
    async with websockets.connect(hub_url, ssl=ssl_context, extra_headers=headers) as websocket:
        # Send the startup sequence
        if startup_sequence:
            logging.info(f"Sending startup sequence {startup_sequence}")
            for message in startup_sequence:
                logging.debug(f"Sending startup sequence message: {message}")
                await websocket.send(json.dumps(message))
        else:
            logging.info("No startup sequence found.")
        async for message in websocket:
            logging.debug(f"Received message: {message}")
