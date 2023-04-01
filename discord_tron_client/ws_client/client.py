import json
import ssl
import websockets

async def websocket_client(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    hub_url = config['hub_url']
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_verify_locations(config['server_cert_path'])

    async with websockets.connect(hub_url, ssl=ssl_context) as websocket:
        async for message in websocket:
            print(f"Received message: {message}")
