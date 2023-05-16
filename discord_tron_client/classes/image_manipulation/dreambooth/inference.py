from accelerate import Accelerator
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch, xformers
torch.manual_seed(420)
# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "junglerally/digital-diffusion"
checkpoints = [ "1500" ]
for checkpoint in checkpoints:
    if checkpoint != "0":
        unet = UNet2DConditionModel.from_pretrained(f"/notebooks/images/datasets/models/checkpoint-{checkpoint}/unet")
        # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        text_encoder = CLIPTextModel.from_pretrained(f"/notebooks/images/datasets/models/checkpoint-{checkpoint}/text_encoder")
        pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder)
    else:
        pipeline = DiffusionPipeline.from_pretrained(model_id)
    pipeline.to("cuda")

    # Perform inference, or save, or push to the hub
    pipeline.save_pretrained("dreambooth-pipeline")
    negative = "low quality, low res, messy, grainy, smooth, sand, big eyes, anime, fractured, cracked, wrinkles, makeup (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, synthetic, rendering"
    output = pipeline(negative_prompt=negative, prompt="a puppy, hanging out on the beach", num_inference_steps=35).images[0]

    output.save(f'/notebooks/test1-{checkpoint}.png')
    output = pipeline(negative_prompt=negative, prompt="a woman, hanging out on the beach", num_inference_steps=35).images[0]

    output.save(f'/notebooks/test2-{checkpoint}.png')
    output = pipeline(negative_prompt=negative, prompt="a woman with areolas, hanging out on the beach", num_inference_steps=35).images[0]

    output.save(f'/notebooks/test3-{checkpoint}.png')