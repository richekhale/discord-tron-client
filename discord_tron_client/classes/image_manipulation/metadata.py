# An image metadata class to handle serialisation of user_config into a JSON object.
# This JSON object will be stored as metadata using PngInfo.

import json, base64, logging
from io import BytesIO
from PIL import Image, PngImagePlugin

class ImageMetadata:
    @staticmethod
    def encode(image: Image, user_config: dict, attributes: dict = None):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("user_config", json.dumps(user_config))
        if attributes is not None:
            # Random attributes can be added to the image, eg. "prompt", "user_id", "user_name"
            for key, value in attributes.items():
                metadata.add_text(key, value)
        # Save into a buffer:
        buffered = BytesIO()
        image.save(buffered, format="PNG", pnginfo=metadata)
        # Retrieve the png with metadata back as an Image:
        image = Image.open(buffered)
        return image
        
    @staticmethod
    def decode(image: Image):
        metadata = image.info
        user_config = json.loads(metadata["user_config"])
        # remove user_config from metadata:
        del metadata["user_config"]
        return user_config, metadata