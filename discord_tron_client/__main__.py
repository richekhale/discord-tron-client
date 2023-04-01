import asyncio
from .ws_client import websocket_client
import logging
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()


async def main():
    logging.info("Starting WebSocket client...")
    await websocket_client(config)

if __name__ == '__main__':
    try:
        # Detect an expired token.
        logging.info("Inspecting auth ticket...")
        from discord_tron_client.classes.auth import Auth
        current_ticket = config.get_auth_ticket()
        auth = Auth(config, current_ticket["access_token"], current_ticket["refresh_token"], current_ticket["expires_in"], current_ticket["issued_at"])
        try:
            is_expired = auth.is_token_expired()
        except Exception as e:
            logging.error(f"Error checking token expiration: {e}")
            is_expired = True
        if is_expired:
            logging.warning("Access token is expired. Attempting to refresh...")
            new_ticket = auth.refresh_client_token(current_ticket["refresh_token"])
            import json
            print(f"New ticket: {json.dumps(new_ticket, indent=4)}")

        # Start the WebSocket client in the background
        asyncio.get_event_loop().run_until_complete(websocket_client(config))

        # Start the Flask server
        from discord_tron_client.app_factory import create_app
        app = create_app()
        app.run()
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        exit(0)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        exit(1)