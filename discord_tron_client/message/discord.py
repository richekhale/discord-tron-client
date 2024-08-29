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
        image_prompt: str = None,
        image_model: str = None,
        user_id: int = None,
    ):
        self.websocket = websocket
        if isinstance(context, DiscordMessage):
            # Extract the context from the existing DiscordMessage
            context = context.data
            logging.debug(f"Extracted data from the DiscordMessage context: {context}")
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
        if image_prompt is not None:
            arguments["image_prompt"] = image_prompt
        if image_model is not None:
            arguments["image_model"] = image_model
        if user_id is not None:
            arguments["user_id"] = user_id
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
    def print_prompt(payload, execute_duration="unknown", attributes: Dict = None):
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
        last_modified = "unknown date"
        if attributes and "last_modified" in attributes:
            last_modified = attributes.get("last_modified", last_modified)
        latest_hash = "unknown hash"
        if attributes and "latest_hash" in attributes:
            latest_hash = attributes.get("latest_hash", latest_hash)

        latent_refiner = "Off"
        latent_refiner_enabled = False
        if "latent_refiner" in user_config and user_config.get("latent_refiner"):
            latent_refiner = "On"
            latent_refiner_enabled = True
        if "refiner_strength" in user_config:
            refiner_strength = str(user_config.get("refiner_strength"))
        if "refiner_guidance" in user_config:
            refiner_guidance = str(user_config.get("refiner_guidance"))
        refiner_guidance_rescale = str(user_config.get("refiner_guidance_rescale", 0.7))
        if "aesthetic_score" in user_config:
            aesthetic_score = str(user_config.get("aesthetic_score"))
        if "negative_aesthetic_score" in user_config:
            negative_aesthetic_score = str(user_config.get("negative_aesthetic_score"))
        if "refiner_strength" in user_config:
            refiner_strength = str(user_config.get("refiner_strength"))
        stage1_guidance = ""
        if "stage1" in user_config.get("model") or "stage-1" in user_config.get(
            "model"
        ):
            stage1_guidance = f"\n**Stage 2 Guidance**: `!settings refiner_guidance {refiner_guidance}`"
        
        flux_adapter = user_config.get('flux_adapter_1')
        if "black-forest-labs" in model_id and flux_adapter:
            model_id = f"!model {model_id}`, **Flux Capacitor** `!settings flux_adapter_1 {flux_adapter}`"

        guidance_rescale = user_config.get("guidance_rescale")
        if latent_refiner == "On":
            latent_refiner = f"{latent_refiner}, `!settings refiner_strength {refiner_strength}` ({float(refiner_strength) * float(steps)}), `!settings refiner_guidance {refiner_guidance}`, `!settings aesthetic_score {aesthetic_score}`, `!settings negative_aesthetic_score {negative_aesthetic_score}`, `!settings refiner_guidance_rescale {refiner_guidance_rescale}`"
        if (
            model_id == "ptx0/s1" or model_id == "ptx0/sdxl-base"
        ) and latent_refiner == "Off":
            model_id = "SDXL Base"
        elif (
            model_id == "ptx0/s1" or model_id == "ptx0/sdxl-base"
        ) and latent_refiner != "Off":
            model_id = "SDXL Base + Refiner"
        else:
            model_id = f"!model {model_id}"
        vmem = 0
        if (
            type(system_hw["video_memory_amount"]) is not str
            or system_hw["video_memory_amount"].isnumeric()
        ):
            vmem = int(system_hw["video_memory_amount"])
        if type(execute_duration) is str and not execute_duration.isdigit():
            execute_time = execute_duration
        else:
            execute_time = round(execute_duration, 2)
        resolution_string = f"{resolution['width']}x{resolution['height']}"
        refiner_status = ""
        if latent_refiner_enabled:
            refiner_status = f"**SDXL Refiner**: {latent_refiner}\n"
        truncate_suffix = ""
        if len(prompt) > 255:
            truncate_suffix = "..(truncated).."
        try:
            return (
                f"<@{author_id}>\n"
                f"**Prompt**: {prompt[:255]}{truncate_suffix}\n"
                f"**Settings**: `!seed {seed}`, `!guidance {user_config['guidance_scaling']}`, `!guidance_rescale {guidance_rescale}`, `!steps {steps}`, `!strength {strength}`, `!resolution {resolution_string}`{stage1_guidance}\n"
                f"**Model**: `{model_id}` (`{latest_hash}` {last_modified})\n{refiner_status}"
                f"**{HardwareInfo.get_identifier()}**: {payload['gpu_power_consumption']}W power used in {execute_time} seconds via {system_hw['gpu_type']} ({vmem}G)\n"  # , on a {system_hw['cpu_type']} with {system_hw['memory_amount']}G RAM\n"
                # f"**Job ID:** `{payload['job_id']}`\n"
            )
        except Exception as e:
            return f"Error generating prompt configuration: {e}"

    @staticmethod
    def mention(payload):
        """Create a Discord mention string from a payload

        Args:
            payload (dict): A dictionary from a Job.

        Returns:
            str: Discord mention string
        """
        if (
            "overridden_user_id" in payload
            and payload["overridden_user_id"] is not None
        ):
            return f"<@{payload['overridden_user_id']}>"
        return f"<@{payload['discord_context']['author']['id']}>"
