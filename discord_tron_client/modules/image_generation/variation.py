from discord_tron_client.classes.image_manipulation import diffusion, pipeline, resolution
from discord_tron_client.classes.image_manipulation.model_manager import TransformerModelManager
from discord_tron_client.message.discord import DiscordMessage
import tqdm, logging, asyncio, io, base64
from PIL import Image
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()
from discord_tron_client.classes.debug import clean_traceback
# Image generator plugin for the worker.
async def variate_image(payload, websocket):
    # We extract the features from the payload and pass them onto the actual generator
    user_config = payload["config"]
    prompt = payload["image_prompt"]
    #model_id = user_config["model"]
    model_id="runwayml/stable-diffusion-v1-5c"
    resolution = user_config["resolution"]
    negative_prompt = user_config["negative_prompt"]
    steps = user_config["steps"]
    positive_prompt = user_config["positive_prompt"]
    discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="edit", message="Prepare for greatness!")
    try:
        await websocket.send(discord_msg.to_json())
        model_manager = TransformerModelManager()
        pipeline_manager = diffusion.DiffusionPipelineManager()
        pipeline_runner = pipeline.PipelineRunner(model_manager=model_manager, pipeline_manager=pipeline_manager, app_config=config, user_config=user_config, discord_msg=discord_msg, websocket=websocket)

        logging.info("Generating image!")
        # Grab the image via http:
        import requests
        image = Image.open(io.BytesIO(requests.get(payload["image_data"]).content))
        result = await pipeline_runner.generate_image(model_id=model_id, prompt=prompt, side_x=resolution["width"], side_y=resolution["height"], negative_prompt=negative_prompt, steps=steps, image=image, promptless_variation=True)
        payload["seed"] = pipeline_runner.seed
        logging.info("Image generated successfully!")
        discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="send", message=DiscordMessage.print_prompt(payload), image=result)
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_context"], module_command="delete")
        await websocket.send(discord_msg.to_json())

    except Exception as e:
        import traceback
        logging.error(f"Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}")
        discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_context"], module_command="delete_errors")
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="edit", message=f"It seems we had an error while generating this image!\n```{e}\n{clean_traceback(traceback.format_exc())}\n```")
        await websocket.send(discord_msg.to_json())
        discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_context"], module_command="delete")
        await websocket.send(discord_msg.to_json())
        raise e