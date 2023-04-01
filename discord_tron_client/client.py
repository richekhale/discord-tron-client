import asyncio
from .ws_client import websocket_client
from .client import app

async def main():
    config_file = "/config/client_config.json"
    await websocket_client(config_file)

if __name__ == '__main__':
    # Start the WebSocket client in the background
    hub_url = "ws://localhost:6789"  # Update this with the WebSocket hub's URL

    asyncio.get_event_loop().run_until_complete(websocket_client(hub_url))
    
    # Start the Flask server
    app.run()
    asyncio.run(main())

