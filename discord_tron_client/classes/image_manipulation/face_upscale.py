import torch, os
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
from RealESRGAN.utils import split_image_into_overlapping_patches, stich_together, unpad_image
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()
scale = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_upscaler(scale: int = 4):
    model_path = config.get_huggingface_model_path()
    model = RealESRGAN(device, scale=4)
    model.load_weights(os.path.join(model_path, 'RealESRGAN_x4.pth'), download=True)
    return model

def use_upscaler(model: RealESRGAN, image: Image):
    sr_image = predict(model, image)
    return sr_image

def pad_reflect(image, pad_size):
    imsize = image.shape
    height, width = imsize[:2]
    new_img = np.zeros([height+pad_size*2, width+pad_size*2, imsize[2]]).astype(np.uint8)
    new_img[pad_size:-pad_size, pad_size:-pad_size, :] = image
    
    new_img[0:pad_size, pad_size:-pad_size, :] = np.flip(image[0:pad_size, :, :], axis=0) #top
    new_img[-pad_size:, pad_size:-pad_size, :] = np.flip(image[-pad_size:, :, :], axis=0) #bottom
    new_img[:, 0:pad_size, :] = np.flip(new_img[:, pad_size:pad_size*2, :], axis=1) #left
    new_img[:, -pad_size:, :] = np.flip(new_img[:, -pad_size*2:-pad_size, :], axis=1) #right
    
    return new_img

@torch.cuda.amp.autocast()
def predict(model, lr_image, batch_size=4, patches_size=192,
            padding=24, pad_size=15):
    lr_image = np.array(lr_image)
    lr_image = pad_reflect(lr_image, pad_size)
    patches, p_shape = split_image_into_overlapping_patches(
        lr_image, patch_size=patches_size, padding_size=padding
    )
    img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()

    with torch.no_grad():
        res = model(img[0:batch_size])
        for i in range(batch_size, img.shape[0], batch_size):
            res = torch.cat((res, model(img[i:i+batch_size])), 0)

    sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
    np_sr_image = sr_image.numpy()

    padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
    scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
    np_sr_image = stich_together(
        np_sr_image, padded_image_shape=padded_size_scaled, 
        target_shape=scaled_image_shape, padding_size=padding * scale
    )
    sr_img = (np_sr_image*255).astype(np.uint8)
    sr_img = unpad_image(sr_img, pad_size*scale)
    sr_img = Image.fromarray(sr_img)

    return sr_img