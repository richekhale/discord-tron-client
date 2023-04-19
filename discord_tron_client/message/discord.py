from discord_tron_client.classes.message import WebsocketMessage
from typing import Dict
from PIL import Image
import logging, websocket, gzip, base64
from io import BytesIO
from discord_tron_client.classes.hardware import HardwareInfo
from discord_tron_client.classes.app_config import AppConfig
hardware = HardwareInfo()
config = AppConfig()
class DiscordMessage(WebsocketMessage):
    def __init__(self, websocket: websocket,  context, module_command: str = "send", mention: str = None, message: str = None, name: str = None, image: Image = None):
        self.websocket = websocket
        if isinstance(context, DiscordMessage):
            # Extract the context from the existing DiscordMessage
            context = context.data
            logging.info(f"Extracted data from the DiscordMessage context: {context}")
        self.context = context
        arguments = {}
        if message is not None:
            arguments["message"] = message
        if name is not None:
            arguments["name"] = name
        if image is not None:
            arguments["image"] = self.b64_image(image)
        if mention is not None:
            arguments["mention"] = mention
        super().__init__(message_type="discord", module_name="message", module_command=module_command, data=context, arguments=arguments)
        
    def b64_image(self, image: Image):
        # Save image to buffer before encoding as base64:

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # Compress the base64-encoded image using gzip
        compressed_b64 = BytesIO()
        with gzip.GzipFile(fileobj=compressed_b64, mode="wb") as gzip_file:
            gzip_file.write(b64_image)
        compressed_b64 = compressed_b64.getvalue().decode('utf-8')
        return compressed_b64

    @staticmethod
    def print_prompt(payload):
        system_hw = hardware.get_machine_info()
        user_config = payload["config"]
        prompt = payload["image_prompt"]
        model_id = user_config["model"]
        resolution = user_config["resolution"]
        steps = user_config["steps"]
        temperature = user_config["temperature"]
        strength = user_config["strength"]
        seed = payload["seed"]
        negative_prompt = user_config["negative_prompt"]
        positive_prompt = user_config["positive_prompt"]
        author_id = payload["discord_context"]["author"]["id"]
        vmem = int(system_hw['video_memory_amount'])
        return f"**<@{author_id}>'s Prompt**: {prompt}\n" \
                f"**Seed**: `!seed {seed}`, **Guidance**: {user_config['guidance_scaling']}, **SAG**: {user_config['enable_sag']}, **SAG-Scale**: {user_config['sag_scale']}\n" \
                f"**Steps**: `!steps {steps}`, **Strength (img2img)**: {strength}, **Temperature (txt2txt)**: {temperature}\n" \
                f"**Model**: `!model {model_id}`\n" \
                f"**Resolution (txt2img)**: " + str(resolution["width"]) + "x" + str(resolution["height"]) + "\n" \
                f"**{config.get_friendly_name() or hardware.get_system_hostname()}**: {payload['gpu_power_consumption']}W power used via {system_hw['gpu_type']} ({vmem}G), on a {system_hw['cpu_type']} with {system_hw['memory_amount']}G RAM\n"
                