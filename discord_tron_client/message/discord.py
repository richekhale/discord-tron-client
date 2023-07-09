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
    def __init__(
        self,
        websocket: websocket,
        context,
        module_command: str = "send",
        mention: str = None,
        message: str = None,
        name: str = None,
        image: Image = None,
        image_url: str = None,
        image_url_list: list = None,
        audio_url: str = None,
        audio_data: str = None,
    ):
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
        if image_url is not None:
            arguments["image_url"] = str(image_url)
        if mention is not None:
            arguments["mention"] = mention
        if image_url_list is not None:
            arguments["image_url_list"] = image_url_list
        if audio_url is not None:
            arguments["audio_url"] = audio_url
        if audio_data is not None:
            arguments["audio_data"] = audio_data
        super().__init__(
            message_type="discord",
            module_name="message",
            module_command=module_command,
            data=context,
            arguments=arguments,
        )

    def b64_image(self, image: Image):
        # Save image to buffer before encoding as base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        # Base64 encode the compressed bytes
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return b64_image

    @staticmethod
    def print_prompt(payload, execute_duration = "unknown"):
        system_hw = hardware.get_machine_info()
        user_config = payload["config"]
        scheduler_config = payload["scheduler_config"]
        scheduler_name = scheduler_config["name"]
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
        
        latent_refiner = "Off"
        if "latent_refiner" in user_config and user_config.get('latent_refiner'):
            latent_refiner = "On"
        if "refiner_strength" in user_config:
            refiner_strength = str(user_config.get('refiner_strength'))
        if "refiner_steps" in user_config:
            refiner_steps = str(user_config.get('refiner_steps'))
        if "refiner_guidance" in user_config:
            refiner_guidance = str(user_config.get('refiner_guidance'))
        if "aesthetic_score" in user_config:
            aesthetic_score = str(user_config.get('aesthetic_score'))
        if "negative_aesthetic_score" in user_config:
            negative_aesthetic_score = str(user_config.get('negative_aesthetic_score'))
        if "refiner_strength" in user_config:
            refiner_strength = str(user_config.get('refiner_strength'))
        guidance_rescale = user_config.get("guidance_rescale")
        if latent_refiner == "On":
            latent_refiner = f"{latent_refiner}, `!settings refiner_strength {refiner_strength}`, `!settings refiner_steps {refiner_steps}`, `!settings refiner_guidance {refiner_guidance}`, `!settings aesthetic_score {aesthetic_score}`, `!settings negative_aesthetic_score {negative_aesthetic_score}`"
        if model_id == "ptx0/s1" and latent_refiner == "Off":
            model_id = "SDXL Base"
        elif model_id == "ptx0/s1" and latent_refiner != "Off":
            model_id = "SDXL Base + Refiner"
        else:
            model_id = f"!model {model_id}"
        vmem = int(system_hw["video_memory_amount"])
        if type(execute_duration) is str and not execute_duration.isdigit():
            execute_time = execute_duration
        else:
            execute_time = round(execute_duration, 2)
        return (
            f"**<@{author_id}>'s Prompt**: {prompt}\n"
            f"**Seed**: `!seed {seed}`, `!guidance {user_config['guidance_scaling']}`, `!settings guidance_rescale {guidance_rescale}`, `!steps {steps}`, `!settings strength {strength}`\n"
            f"**Model**: `{model_id}`\n"
            f"**SDXL Refiner**: {latent_refiner}\n"
            f"**Resolution**: "
            + str(resolution["width"])
            + "x"
            + str(resolution["height"])
            + "\n"
            f"**{HardwareInfo.get_identifier()}**: {payload['gpu_power_consumption']}W power used in {execute_time} seconds via {system_hw['gpu_type']} ({vmem}G), on a {system_hw['cpu_type']} with {system_hw['memory_amount']}G RAM\n"
        )

    @staticmethod
    def mention(payload):
        """Create a Discord mention string from a payload

        Args:
            payload (dict): A dictionary from a Job.

        Returns:
            str: Discord mention string
        """
        return f"<@{payload['discord_context']['author']['id']}>"
