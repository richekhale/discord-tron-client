from discord_tron_client.classes.message import WebsocketMessage
from typing import Dict
from PIL import Image
import logging, websocket
import base64
from io import BytesIO
from discord_tron_client.classes.hardware import HardwareInfo
hardware = HardwareInfo()

class DiscordMessage(WebsocketMessage):
    def __init__(self, websocket: websocket,  context, module_command: str = "send", message: str = "Loading the model and preparing to generate your image!", image: Image = None):
        self.websocket = websocket
        if isinstance(context, DiscordMessage):
            # Extract the context from the existing DiscordMessage
            context = context.data
            logging.info(f"Extracted data from the DiscordMessage context: {context}")
        self.context = context
        arguments = { "message": message }
        if image is not None:
            arguments["image"] = self.b64_image(image)
        super().__init__(message_type="discord", module_name="message", module_command=module_command, data=context, arguments=arguments)
        
    def b64_image(self, image: Image):
        # Save image to buffer before encoding as base64:

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return b64_image
    
    @staticmethod
    def print_prompt(payload):
        system_hw = hardware.get_machine_info()
        user_config = payload["config"]
        prompt = payload["image_prompt"]
        model_id = user_config["model"]
        resolution = user_config["resolution"]
        negative_prompt = user_config["negative_prompt"]
        steps = user_config["steps"]
        temperature = user_config["temperature"]
        strength = user_config["strength"]
        positive_prompt = user_config["positive_prompt"]
        vmem = int(system_hw['video_memory_amount'])
        return f"**Prompt**: {prompt}\n" \
                f"**Steps**: {steps}, **Strength (img2img)**: {strength}, **Temperature (txt2txt)**: {temperature}\n" \
                f"**Model**: {model_id}\n" \
                f"**Resolution (txt2img)**: " + str(resolution["width"]) + "x" + str(resolution["height"]) + "\n" \
                f"**{hardware.get_system_hostname()}**: {system_hw['gpu_type']} ({vmem}G), on a {system_hw['cpu_type']} with {system_hw['memory_amount']}G RAM\n"
                