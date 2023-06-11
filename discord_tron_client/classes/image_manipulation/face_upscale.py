import torch, os
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_upscaler(scale: int = 4):
    model_path = config.get_huggingface_model_path()
    model = RealESRGAN(device, scale=4)
    model.load_weights(os.path.join(model_path, 'RealESRGAN_x4.pth'), download=True)
    return model

def use_upscaler(model: RealESRGAN, image: Image):
    sr_image = model.predict(image)
    return sr_image