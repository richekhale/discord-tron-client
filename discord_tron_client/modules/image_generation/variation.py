from discord_tron_client.classes.image_manipulation import (
    diffusion,
    pipeline,
    resolution,
)
from discord_tron_client.classes.image_manipulation.model_manager import (
    TransformerModelManager,
)
from discord_tron_client.message.discord import DiscordMessage
from discord_tron_client.classes.uploader import Uploader
import tqdm, logging, asyncio, io, base64
from PIL import Image
from discord_tron_client.classes.app_config import AppConfig

config = AppConfig()
from discord_tron_client.classes.debug import clean_traceback


# Image generator plugin for the worker.
async def promptless_variation(payload, websocket):
    # We extract the features from the payload and pass them onto the actual generator
    user_config = payload["config"]
    scheduler_config = payload["scheduler_config"]
    prompt = payload["image_prompt"]
    model_id = user_config["model"]
    # model_id = "lambdalabs/sd-image-variations-diffusers"
    resolution = user_config["resolution"]
    negative_prompt = user_config["negative_prompt"]
    steps = user_config["steps"]
    positive_prompt = user_config["positive_prompt"]
    controlnet_models_broken = [
        'stabilityai/stable-diffusion-2',
        'stabilityai/stable-diffusion-2-1',
        'junglerally/digital-diffusion',
        'ptx0/artius_v21',
        'stablediffusionapi/illuminati-diffusion',
        'ptx0/realism-engine',
        'ptx0/sdxl-base',
        'ptx0/s1',
        'ptx0/s2'
    ]
    controlnet_warning = ""
    default_controlnet_model = "theintuitiveye/HARDblend"
    if model_id.lower() in controlnet_models_broken:
        controlnet_warning = f" Your model `{model_id}` was not in our compatibility list for ControlNet. It has been swapped to `{default_controlnet_model}`!\n" \
                                f"The following models are currently not supported for ControlNet: {', '.join(controlnet_models_broken)}"
        model_id = default_controlnet_model
    discord_msg = DiscordMessage(
        websocket=websocket,
        context=payload["discord_first_message"],
        module_command="edit",
        message=f"Beginning work on your 🍕 💩 image variation!{controlnet_warning}",
    )
    try:
        websocket = AppConfig.get_websocket()
        await websocket.send(discord_msg.to_json())
        model_manager = TransformerModelManager()
        pipeline_manager = AppConfig.get_pipeline_manager()
        pipeline_runner = pipeline.PipelineRunner(
            model_manager=model_manager,
            pipeline_manager=pipeline_manager,
            app_config=config,
            user_config=user_config,
            discord_msg=discord_msg,
            websocket=websocket,
        )

        logging.info("Generating image!")
        # Grab the image via http:
        import requests

        image = Image.open(
            io.BytesIO(requests.get(payload["image_data"], timeout=10).content)
        )
        from discord_tron_client.classes.image_manipulation.image_tiler import (
            ImageTiler,
        )

        image = image.resize(
            (resolution["width"], resolution["height"]), resample=Image.LANCZOS
        )
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_context"],
            module_command="delete",
        )
        await websocket.send(discord_msg.to_json())
        # Grab starting timestamp
        user_config["user_id"] = payload["discord_context"]["author"]["id"]
        start_time = asyncio.get_running_loop().time()
        result = await pipeline_runner.generate_image(
            user_config=user_config,
            scheduler_config=scheduler_config,
            model_id=model_id,
            prompt=prompt,
            side_x=resolution["width"],
            side_y=resolution["height"],
            negative_prompt=negative_prompt,
            steps=steps,
            image=image,
            promptless_variation=True,
        )
        end_time = asyncio.get_running_loop().time()
        total_time = end_time - start_time
        payload["seed"] = pipeline_runner.seed
        payload["gpu_power_consumption"] = pipeline_runner.gpu_power_consumption
        websocket = AppConfig.get_websocket()
        logging.info("Image generated successfully!")
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="delete",
        )
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="send",
            message=DiscordMessage.print_prompt(payload, execute_duration=total_time),
            image=result,
        )
        await websocket.send(discord_msg.to_json())

    except Exception as e:
        import traceback

        logging.error(
            f"Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}"
        )
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_context"],
            module_command="delete_errors",
        )
        websocket = AppConfig.get_websocket()
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="edit",
            message=f"It seems we had an error while generating this image!\n```{e}\n{clean_traceback(traceback.format_exc())}\n```",
        )
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_context"],
            module_command="delete",
        )
        await websocket.send(discord_msg.to_json())
        raise e

def round_to_nearest_multiple(value, multiple):
    """Round a value to the nearest multiple."""
    rounded = round(value / multiple) * multiple
    return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'

from math import sqrt

def calculate_new_size_by_pixel_area(W: int, H: int, megapixels: float):
    aspect_ratio = W / H
    total_pixels = megapixels * 1e6  # Convert megapixels to pixels

    W_new = int(round(sqrt(total_pixels * aspect_ratio)))
    H_new = int(round(sqrt(total_pixels / aspect_ratio)))

    # Ensure they are divisible by 8
    W_new = round_to_nearest_multiple(W_new, 64)
    H_new = round_to_nearest_multiple(H_new, 64)

    return W_new, H_new

async def prompt_variation(payload, websocket):
    # We extract the features from the payload and pass them onto the actual generator
    user_config = payload["config"]
    scheduler_config = payload["scheduler_config"]
    prompt = payload["image_prompt"]
    model_id = user_config["model"]
    if 'ptx0' not in user_config["model"]:
        model_id = "ptx0/sdxl-base"
    resolution = user_config["resolution"]
    negative_prompt = user_config["negative_prompt"]
    steps = user_config["steps"]
    positive_prompt = user_config["positive_prompt"]
    discord_msg = DiscordMessage(
        websocket=websocket,
        context=payload["discord_first_message"],
        module_command="edit",
        message=f"{DiscordMessage.mention(payload)} Beginning work on your image variation!",
    )
    try:
        websocket = AppConfig.get_websocket()
        await websocket.send(discord_msg.to_json())
        model_manager = TransformerModelManager()
        pipeline_manager = AppConfig.get_pipeline_manager()
        pipeline_runner = pipeline.PipelineRunner(
            model_manager=model_manager,
            pipeline_manager=pipeline_manager,
            app_config=config,
            user_config=user_config,
            discord_msg=discord_msg,
            websocket=websocket,
        )

        logging.info("Generating image!")
        # Grab the image via http:
        import requests

        image = Image.open(
            io.BytesIO(requests.get(payload["image_data"], timeout=10).content)
        )
        factor = 1.0
        # See if the prompt has a `--upscale` parameter and then multiply it by a maximum of 2 or factor to upscale
        if "--upscale" in prompt:
            upscale_factor = float(prompt.split("--upscale")[1].split(" ")[0])
            factor = min(upscale_factor, 2.0)
            # Remove --upscale from prompt:
            prompt = prompt.split("--upscale")[0]
        new_width, new_height = calculate_new_size_by_pixel_area(image.width, image.height, factor)
        image = image.resize(
            (new_width, new_height), resample=Image.LANCZOS
        )
        
        try:
            background = Image.new("RGBA", image.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, image)
            image = alpha_composite.convert("RGB")
        except Exception as e:
            logging.error(f"Error compositing image: {e}")
            alpha_composite = image
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_context"],
            module_command="delete",
        )
        await websocket.send(discord_msg.to_json())
        # Grab starting timestamp
        user_config["user_id"] = payload["discord_context"]["author"]["id"]
        start_time = asyncio.get_running_loop().time()
        output_images = await pipeline_runner.generate_image(
            user_config=user_config,
            scheduler_config=scheduler_config,
            prompt=prompt + " " + positive_prompt,
            model_id=model_id,
            side_x=resolution["width"],
            side_y=resolution["height"],
            negative_prompt=negative_prompt,
            steps=steps,
            image=image,
            prompt_variation=True,
        )
        end_time = asyncio.get_running_loop().time()
        total_time = end_time - start_time
        payload["seed"] = pipeline_runner.seed
        payload["gpu_power_consumption"] = pipeline_runner.gpu_power_consumption
        websocket = AppConfig.get_websocket()
        logging.info("Image generated successfully!")
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="edit",
            message=f"{DiscordMessage.mention(payload)} Uploading your image variants!",
        )
        await websocket.send(discord_msg.to_json())

        api_client = AppConfig.get_api_client()
        uploader = Uploader(api_client=api_client, config=config)
        url_list = await uploader.upload_images(output_images)
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="delete",
        )
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="send",
            message=DiscordMessage.print_prompt(payload, execute_duration=total_time),
            image_url_list=url_list,
        )
        await websocket.send(discord_msg.to_json())

    except Exception as e:
        import traceback

        logging.error(
            f"Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}"
        )
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_context"],
            module_command="delete_errors",
        )
        websocket = AppConfig.get_websocket()
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="edit",
            message=f"It seems we had an error while generating this image!\n```{e}\n{clean_traceback(traceback.format_exc())}\n```",
        )
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_context"],
            module_command="delete",
        )
        await websocket.send(discord_msg.to_json())
        raise e
