import torch, logging
from PIL import Image
from diffusers import ControlNetModel, DiffusionPipeline
from diffusers.utils import load_image
from transformers import AutoModel

# from huggingface_hub import login as hf_login

# hf_login()

from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

logger = logging.getLogger()
logger.setLevel("DEBUG")

torch_backend = "cuda"
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        logging.warning(
            "Apple Silicon Users: MPS not available because the current PyTorch install was not "
            "built with MPS enabled. This warning can be disregarded for anyone else."
        )
    else:
        logging.warning(
            "Apple Silicon Users: MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine. This warning can be disregarded."
        )
else:
    torch_backend = "mps"
device = torch.device(torch_backend)
# stage 1
logging.debug(f"Loading DeepFloyd Stage1 model.")
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
)
logging.debug(f"Enable DeepFloyd model CPU offload.")
stage_1.enable_model_cpu_offload()

logging.debug(f"Using DeepFloyd Stage2")
deepfloyd_stage2 = True
# prompt = "a stunning and impossible caustics experiment, suspended liquids, amorphous liquid forms, high intensity light rays, unreal engine 5, raytracing, 4k, laser dot fields, curving light energy beams, glowing energetic caustic liquids, thousands of prismatic bubbles, quantum entangled light rays from other dimensions, negative width height, recursive dimensional portals"
generator = torch.Generator(device=device)
generator.manual_seed(int(0))


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


### Begin Image Generation ###


def generate(
    prompt="a portrait of a cute smiling dog with one ear up and one ear down, with a collar that says the number '4', bokeh, depth of field",
    deepfloyd_stage2=True,
    negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
    width=64,
    height=64,
):
    logging.debug(f"Generating prompt embeds.")
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt, negative_prompt)
    logging.debug(f"Generating stage 1 output.")
    image = stage_1(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        output_type="pt",
        width=width,
        height=height,
    ).images
    stage1img = pt_to_pil(image)[0]
    s2_width = stage1img.width * 4
    s2_height = stage1img.height * 4
    s3_width = s2_width * 4
    s3_height = s2_height * 4
    stage1img.save("/notebooks/if_stage_I.png")
    if deepfloyd_stage2:
        logging.debug(f"Using DF Stage 2 for next step...")
        # stage 2
        stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        # stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        stage_2.enable_model_cpu_offload()
        image = stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
            width=s2_width,
            height=s2_height,
        ).images
        stage2img = pt_to_pil(image)[0]
        stage2img.save("/notebooks/_/if_stage_II.png")
        # stage 3
        safety_modules = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": None,
            "watermarker": None,
        }
        stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            **safety_modules,
            torch_dtype=torch.float16,
        )
        # stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        stage_3.enable_model_cpu_offload()
        image = stage_3(
            prompt=prompt, image=image, generator=generator, noise_level=100
        ).images

    else:
        logging.debug(f"Using ControlNet 1.5 for next step...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16
        )
        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            custom_pipeline="stable_diffusion_controlnet_img2img",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to(torch_backend)
        pipe.enable_xformers_memory_efficient_attention()

        condition_image = resize_for_condition_image(stage1img, 1024)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=condition_image,
            controlnet_conditioning_image=condition_image,
            width=condition_image.size[0],
            height=condition_image.size[1],
            strength=1.0,
            generator=generator,
            num_inference_steps=32,
        ).images[0]

    ### Completed image generation output.. Now save it. ###
    if not deepfloyd_stage2:
        image.save("/notebooks/_/output.png")
    else:
        image[0].save("/notebooks/_/if_stage_III.png")
    logging.debug(f"Saved output.")


if __name__ == "__main__":
    generate()
