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
import tqdm, logging, asyncio
from PIL import Image
from discord_tron_client.classes.app_config import AppConfig

config = AppConfig()
from discord_tron_client.classes.debug import clean_traceback


# Image generator plugin for the worker.
async def generate_image(payload, websocket):
    # We extract the features from the payload and pass them onto the actual generator
    try:
        # Grab a beginning timestamp:
        start_time = asyncio.get_event_loop().time()
        user_config = payload["config"]
        prompt = payload["image_prompt"]
        model_id = user_config["model"]
        resolution = user_config["resolution"]
        negative_prompt = user_config["negative_prompt"]
        steps = user_config["steps"]
        model_config = payload.get("model_config", {})
        positive_prompt = user_config["positive_prompt"]
        upscaler = payload.get("upscaler", False)
        websocket = AppConfig.get_websocket()
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="edit",
            message="Your prompt is now being processed. This might take a while to get to the next step if we have to download your model!",
        )
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
            model_config=model_config,
        )
        # Attach a positive prompt weight to the end so that it's more likely to show up this way.
        prompt = prompt + " " + positive_prompt
        image = None
        if "image_data" in payload:
            logging.debug(f"Found image data in payload: {payload['image_data']}")
            import io, requests

            image = Image.open(
                io.BytesIO(requests.get(payload["image_data"], timeout=10).content)
            )
            image = image.resize((resolution["width"], resolution["height"]), resample=Image.LANCZOS)
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
        if "overridden_user_id" in payload and payload["overridden_user_id"] is not None:
            payload["discord_context"]["author"]["id"] = payload["overridden_user_id"]
        user_config["user_id"] = payload["discord_context"]["author"]["id"]
        if "width" not in resolution or "height" not in resolution:
            resolution = {"width": 1024, "height": 1024}
        output_images = await pipeline_runner.generate_image(
            user_config=user_config,
            prompt=prompt,
            model_id=model_id,
            side_x=resolution["width"],
            side_y=resolution["height"],
            negative_prompt=negative_prompt,
            steps=steps,
            image=image,
            upscaler=upscaler,
        )
        end_time = asyncio.get_event_loop().time()
        
        websocket = AppConfig.get_websocket()
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="delete",
        )
        for attempt in range(1, 6):
            if (
                not websocket
                or not hasattr(websocket, "open")
                or websocket.open != True
            ):
                logging.warn(
                    "WebSocket connection is not open. Retrieving fresh instance."
                )
                websocket = AppConfig.get_websocket()
                await asyncio.sleep(2)
            else:
                logging.debug("WebSocket connection is open. Continuing.")
                break
        await websocket.send(discord_msg.to_json())
        payload["seed"] = pipeline_runner.seed
        payload["gpu_power_consumption"] = pipeline_runner.gpu_power_consumption
        logging.info(
            "Image generated successfully!"
        )  # Truncate prompt to 32 chars and add a ...
        truncated_prompt = prompt[:29] + "..."

        # Try uploading via the HTTP API
        api_client = AppConfig.get_api_client()
        uploader = Uploader(api_client=api_client, config=config)
        url_list = await uploader.upload_images(output_images)
        # Now we can remove the message.
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="delete",
        )
        await websocket.send(discord_msg.to_json())
        # discord_msg = DiscordMessage(websocket=websocket, context=payload["discord_first_message"], module_command="send", message=DiscordMessage.print_prompt(payload), image_url_list=url_list)
        execute_duration = end_time - start_time
        websocket = AppConfig.get_websocket()
        attributes = {
            "last_modified": pipeline_manager.pipeline_versions.get(model_id, {}).get("last_modified", "unknown"),
            "latest_hash": pipeline_manager.pipeline_versions.get(model_id, {}).get('latest_hash', "unknown hash")
        }
        discord_msg = DiscordMessage(
            websocket=websocket,
            context=payload["discord_first_message"],
            module_command="create_thread",
            name=truncated_prompt,
            image_model=model_id,
            image_prompt=prompt,
            message=DiscordMessage.print_prompt(payload, execute_duration=execute_duration, attributes=attributes),
            image_url_list=url_list,
            user_id=payload["discord_context"]["author"]["id"],
        )
        await websocket.send(discord_msg.to_json())

    except Exception as e:
        import traceback

        try:
            s = str(e)
            if "out of memory" in s:
                logging.error("The exception occurred because we ran out of memory. Clearing CUDA.")
                import torch, gc
                torch.cuda.empty_cache()
                gc.collect()
                pipeline_manager.delete_pipes()

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
                message=f"It seems we had an error while generating this image!\n```{e}\n```",
            )
            await websocket.send(discord_msg.to_json())
            discord_msg = DiscordMessage(
                websocket=websocket,
                context=payload["discord_context"],
                module_command="delete",
            )
            await websocket.send(discord_msg.to_json())
            raise e
        except Exception as e_squash:
            logging.error(f"Error squashed: {e}, traceback: {traceback.format_exc()}")
