import logging
from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)


class SdxlRefinerPipelineRunner(BasePipelineRunner):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, **args):
        user_config = args.get("user_config", None)
        del args["user_config"]  # This doesn't get passed to Diffusers.
        logging.debug(f'Args (minus user_config) for SDXL refiner: {args}')
        # Currently, it seems like the refiner's prompt weighting is broken.
        # We are disabling it by default.
        if user_config is not None and user_config.get("refiner_prompt_weighting", True) and config.enable_compel():
            logging.info(f'Using SDXL prompt weighting.')
            # SDXL, when using prompt embeds, must be instructed to only generate 1 image per prompt.
            args["num_images_per_prompt"] = 1
            # We don't pass the text prompts in, when using embeds.
            for unwanted_arg in ["prompt", "negative_prompt"]:
                if unwanted_arg in args:
                    del args[unwanted_arg]
        else:
            logging.info(f'Using prompt and negative_prompt strings.')
            # If we're not using prompt embeds, delete them from the arguments.
            for unwanted_arg in [
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            ]:
                if unwanted_arg in args:
                    del args[unwanted_arg]
            if type(args['prompt'] == list):
                logging.debug(f'Received list prompts. Stripping to a string...')
                args['prompt'] = args['prompt'][0]
                args['negative_prompt'] = args['negative_prompt'][0]
        return_images = []
        processing_images = args['image']
        del args['image']
        for idx, image in range(0, len(processing_images)):
            args['image'] = image
            return_images.append(self.pipeline(**args).images)
        return return_images
