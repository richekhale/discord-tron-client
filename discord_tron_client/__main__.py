import random, asyncio
from .ws_client import websocket_client
import logging
from discord_tron_client.classes import log_format
from discord_tron_client.classes.image_manipulation.diffusion import (
    DiffusionPipelineManager,
)
from discord_tron_client.classes.app_config import AppConfig
from discord_tron_client.classes.api_client import ApiClient

config = AppConfig()
config.set_pipeline_manager(DiffusionPipelineManager())


def main():
    try:
        # Detect an expired token.
        logging.info("Inspecting auth ticket...")
        from discord_tron_client.classes.auth import Auth

        current_ticket = config.get_auth_ticket()
        auth = Auth(
            config,
            current_ticket["access_token"],
            current_ticket["refresh_token"],
            current_ticket["expires_in"],
            current_ticket["issued_at"],
        )
        auth.get()
        api_client = ApiClient(auth=auth, config=config)
        AppConfig.set_api_client(api_client)
        # Start the WebSocket client in the background
        startup_sequence = []
        from discord_tron_client.classes.message import WebsocketMessage

        # Add any startup sequence here
        from discord_tron_client.classes.hardware import HardwareInfo
        from discord_tron_client.classes.image_manipulation.resolution import (
            ResolutionManager,
        )

        hardware_info = HardwareInfo()
        machine_info = hardware_info.get_machine_info()
        identifier = HardwareInfo.get_identifier()
        register_data = hardware_info.get_register_data(worker_id=identifier)
        register_data["hardware"] = hardware_info.get_simple_hardware_info()
        register_data["available_resolutions"] = (
            ResolutionManager.get_resolutions_with_extra_data()
        )
        hello_world_message = WebsocketMessage(
            message_type="hello_world",
            module_name="worker",
            module_command="register",
            arguments=register_data,
        )
        startup_sequence.append(hello_world_message)
        hardware_info_message = WebsocketMessage(
            message_type="hardware_info",
            module_name="system",
            module_command="update",
            arguments=machine_info,
        )
        startup_sequence.append(hardware_info_message)
        main_loop = asyncio.get_event_loop()
        # Add the main loop to the central Config object.
        AppConfig.set_loop(main_loop)
        main_loop.run_until_complete(
            websocket_client(config, startup_sequence, auth=auth)
        )

        # Start the Flask server
        from discord_tron_client.app_factory import create_app

        app = create_app()
        app.run()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        exit(0)
    except Exception as e:
        import traceback

        logging.error(f"Stack trace: {traceback.format_exc()}")
        exit(1)


if __name__ == "__main__":
    main()
