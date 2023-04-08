from discord_tron_client.classes.image_manipulation import diffusion, pipeline, resolution
from discord_tron_client.classes.image_manipulation.model_manager import TransformerModelManager
from discord_tron_client.message.discord import DiscordMessage
import tqdm, logging, asyncio
from discord_tron_client.classes.app_config import AppConfig
config = AppConfig()
from discord_tron_client.classes.debug import clean_traceback
# Image generator plugin for the worker.
async def generate_image(payload, websocket):
    # We extract the features from the payload and pass them onto the actual generator
    user_config = payload["config"]
    prompt = payload["image_prompt"]
    model_id = user_config["model"]
    resolution = user_config["resolution"]
    negative_prompt = user_config["negative_prompt"]
    steps = user_config["steps"]
    positive_prompt = user_config["positive_prompt"]
    discord_msg = DiscordMessage(module_command="edit", message="Prepare for greatness!", context=payload["discord_first_message"])
    #   def __init__(self, model_manager, pipeline_manager, config):
    try:
        send_result = await discord_msg.send(websocket)
        model_manager = TransformerModelManager()
        pipeline_manager = diffusion.DiffusionPipelineManager()
        pipeline_runner = pipeline.PipelineRunner(model_manager, pipeline_manager, config)
        logging.info(f"Sent result: {send_result}")
        result = pipeline_runner.generate_image(prompt, model_id, resolution, negative_prompt, steps, positive_prompt, user_config, discord_msg, websocket)
    except Exception as e:
        import traceback
        logging.error(f"Error generating image: {e}")
        discord_msg = DiscordMessage(module_command="delete_errors", message="", context=payload["discord_context"])
        send_result = await discord_msg.send(websocket)
        discord_msg = DiscordMessage(module_command="edit", message=f"It seems we had an error while generating this image!\n```{e}\n{clean_traceback(traceback.format_exc())}\n```", context=payload["discord_first_message"])
        send_result = await discord_msg.send(websocket)
        discord_msg = DiscordMessage(module_command="delete", message="", context=payload["discord_context"])
        send_result = await discord_msg.send(websocket)
        raise e