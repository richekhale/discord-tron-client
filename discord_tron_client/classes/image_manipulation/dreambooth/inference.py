from accelerate import Accelerator
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
torch.manual_seed(420420420)
# Load the pipeline with the same arguments (model, revision) that were used for training
#model_id = "/notebooks/images/datasets/models"
model_id = "stabilityai/stable-diffusion-2-1"
torch.set_float32_matmul_precision('high')
# Find the latest checkpoint
import os
checkpoints = [ int(x.split('-')[1]) for x in os.listdir(f'/notebooks/images/datasets/models/') if x.startswith('checkpoint-') ]
checkpoints.sort()
range_begin = 0
range_step = 50
range_end = checkpoints[-1]
print(f'Highest checkpoint found so far: {range_end}')

# Convert numeric range to an array of string numerics:
checkpoints = [ str(x) for x in range(range_begin, range_end + range_step, range_step) ]
checkpoints = [ "4000" ]
for checkpoint in checkpoints:
    if len(checkpoints) > 1 and os.path.isfile(f'/notebooks/_/target-{checkpoint}.png'):
        continue
    try:
        print(f'Loading checkpoint: {checkpoint}')
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
    prompts = {
        "woman": "a woman, hanging out on the beach",
        "man": "a man playing guitar in a park",
        "child": "a child flying a kite on a sunny day",
        "alien": "an alien exploring the Mars surface",
        "robot": "a robot serving coffee in a cafe",
        "knight": "a knight protecting a castle",
        "target": "a woman with areolas, standing in a midnight field, surrounded by stars, with the milky way looming overhead",
        "twoget": "a woman with areolas",
        "threget": "areolas",
        "wimmen": "a group of women with areolas",
        "menn": "a group of men",
        "bicycle": "a bicycle, on a mountainside, on a sunny day",
        "cosmic": "cosmic entity, sitting in an impossible position, quantum reality, colours",
        "wizard": "a mage wizard, bearded and gray hair, blue  star hat with wand and mystical haze",
        "wizarddd": "digital art, fantasy, portrait of an old wizard, detailed",
        "macro": "a dramatic city-scape at sunset or sunrise",
        "micro": "RNA and other molecular machinery of life",
        "gecko": "a leopard gecko stalking a cricket"
    }

    for shortname, prompt in prompts.items():
        if not os.path.isfile(f'/notebooks/_/{shortname}-{checkpoint}.png'):
            print(f'Generating {shortname} at {checkpoint}')
            #output = pipeline(negative_prompt=negative, prompt=prompt, num_inference_steps=50).images[0]
            #output.save(f'/notebooks/_/{shortname}-{checkpoint}.png')
        
    if not os.path.exists(f'/notebooks/images/datasets/models/pipeline'):
        print(f'Saving pretrained pipeline.')
        pipeline.save_pretrained('/notebooks/images/datasets/models/beta-v3')
