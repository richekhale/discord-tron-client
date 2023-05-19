from accelerate import Accelerator
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
torch.manual_seed(420420420)
# Load the pipeline with the same arguments (model, revision) that were used for training
#model_id = "/notebooks/images/datasets/models"
model_id = "stabilityai/stable-diffusion-2-1"

# Find the latest checkpoint
import os
checkpoints = [ int(x.split('-')[1]) for x in os.listdir(f'/notebooks/images/datasets/models/') if x.startswith('checkpoint-') ]
checkpoints.sort()
range_begin = 200
range_step = range_begin
range_end = checkpoints[-1]
print(f'Highest checkpoint found so far: {range_end}')

# Convert numeric range to an array of string numerics:
checkpoints = [ str(x) for x in range(range_begin, range_end + range_step, range_step) ]
checkpoints = [ "8800" ]
for checkpoint in checkpoints:
    if len(checkpoints) > 1 and os.path.isfile(f'/notebooks/_/puppy-{checkpoint}.png'):
        continue
    try:
        if checkpoint != "0":
            unet = UNet2DConditionModel.from_pretrained(f"/notebooks/images/datasets/models/checkpoint-{checkpoint}/unet")
            unet = torch.compile(unet)
            # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
            text_encoder = CLIPTextModel.from_pretrained(f"/notebooks/images/datasets/models/checkpoint-{checkpoint}/text_encoder")
            pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder)
        else:
            pipeline = DiffusionPipeline.from_pretrained(model_id)
            pipeline.unet = torch.compile(pipeline.unet)
        pipeline.to("cuda")
    except:
        print(f'Could not generate pipeline for checkpoint {checkpoint}')
        continue
    # Does the file exist already?
    import os
    negative = "low quality, low res, oorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, synthetic, rendering"
    pipeline.save_pretrained('/notebooks/images/datasets/models/beta')
    if not os.path.isfile(f'/notebooks/_/puppy-{checkpoint}.png'):
        print(f'Generating puppy at {checkpoint}')
        output = pipeline(negative_prompt=negative, prompt="a puppy, hanging out on the beach", num_inference_steps=50).images[0]
        output.save(f'/notebooks/_/puppy-{checkpoint}.png')
    if not os.path.isfile(f'/notebooks/_/woman-{checkpoint}.png'):
        print(f'Generating woman at {checkpoint}')
        output = pipeline(negative_prompt=negative, prompt="a school-teacher, hanging out on the beach", num_inference_steps=50).images[0]
        output.save(f'/notebooks/_/woman-{checkpoint}.png')
    if not os.path.isfile(f'/notebooks/_/target-{checkpoint}.png'):
        print(f'Generating target at {checkpoint}.')
        output = pipeline(negative_prompt=negative, prompt="an indian woman, standing outside on a sunny day, smiling, cinematic, 8k, sharp", num_inference_steps=50).images[0]
        output.save(f'/notebooks/_/target-{checkpoint}.png')
    if not os.path.exists(f'/notebooks/images/datasets/models/pipeline'):
        print(f'Saving pretrained pipeline.')
    pipeline.save_pretrained('/notebooks/images/datasets/models/beta-v2')
