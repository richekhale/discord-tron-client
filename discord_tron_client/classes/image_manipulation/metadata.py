# An image metadata class to handle serialisation of user_config into a JSON object.
# This JSON object will be stored as metadata using PngInfo.

import json, base64, logging
from io import BytesIO
from PIL import Image, PngImagePlugin

class ImageMetadata:
    @staticmethod
    def encode(image: Image, user_config: dict, attributes: dict = None):
        metadata = PngImagePlugin.PngInfo()
        # remove gpt_role from metadata, if there:
        user_config_copy = user_config.copy()
        if 'gpt_role' in user_config:
            del user_config_copy['gpt_role']
        metadata.add_text("user_config", json.dumps(user_config_copy))
        metadata.add_text("parameters", ImageMetadata.automatic1111_metadata(user_config_copy, attributes))
        if attributes is not None:
            # Random attributes can be added to the image, eg. "prompt", "user_id", "user_name"
            for key, value in attributes.items():
                metadata.add_text(key, str(value))
        # Save into a buffer:
        buffered = BytesIO()
        image.save(buffered, format="PNG", pnginfo=metadata)
        # Retrieve the png with metadata back as an Image:
        image = Image.open(buffered)
        logging.debug(f'Image metadata: {image.info}')
        return image
        
    @staticmethod
    def decode(image: Image):
        metadata = image.info
        user_config = json.loads(metadata["user_config"])
        # remove user_config from metadata:
        del metadata["user_config"]
        # remove gpt_role from metadata, if there:
        if 'gpt_role' in user_config:
            del user_config['gpt_role']
        return user_config, metadata
    
    @staticmethod
    def automatic1111_metadata(user_config: dict, attributes: dict):
        # We want to allow importing settings into A1111, and export gen data to CivitAI.
        # Example output:
        # attributes.get('prompt', 'no prompt given')
        # Steps: user_config.get('steps', 0), Sampler: DDIM, CFG scale: attributes['guidance'], Seed: attributes['seed'], Size: user_config['width] x user_config['height], Model hash: 12345678
        a1111_output = attributes.get('prompt', user_config.get('positive_prompt', 'no prompt given'))
        a1111_output = f'{a1111_output}\nSteps: {user_config.get("steps", 0)}, Sampler: {attributes.get("scheduler", "DDIM")}, CFG scale: {attributes.get("guidance_scaling", user_config.get("guidance_scaling"))}, Seed: {attributes["seed"]}, Size: {user_config["resolution"]["width"]}x{user_config["resolution"]["height"]}, Model hash: 12345678'

        return a1111_output
