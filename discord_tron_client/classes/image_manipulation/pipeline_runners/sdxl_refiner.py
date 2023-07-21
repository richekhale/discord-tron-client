from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)


class SdxlRefinerPipelineRunner(BasePipelineRunner):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, **args):
        # Set all defaults at once
        default_values = {
            'prompt': None, 'negative_prompt': None, 
            'num_inference_steps': None, 'guidance_scale': None, 
            'guidance_rescale': None, 'num_images_per_prompt': None, 
            'output_type': None, 'aesthetic_score': None, 
            'negative_aesthetic_score': None, 'generator': None, 
            'user_config': None, 'prompt_embeds': None, 
            'negative_prompt_embeds': None, 'pooled_prompt_embeds': None,
            'negative_pooled_prompt_embeds': None, 'strength': None, 
            'image': None, 'denoising_start': None, 'denoising_end': None,
            'width': None, 'height': None
        }
        args = {**default_values, **args}  # merge the two dictionaries, with priority on args
        extra_args = {}
        if args['image'] is not None:
            extra_args['image'] = args['image']
        if args['width'] is not None and args['height'] is not None:
            extra_args['width'] = args['width']
            extra_args['height'] = args['height']
        
        if args['user_config'] is not None and args['user_config'].get("refiner_prompt_weighting", False):
            # Currently, it seems like the refiner's prompt weighting is broken.
            # We are disabling it by default.
            args['num_images_per_prompt'] = 1  # SDXL, when using prompt embeds, only generates 1 image per prompt.
            # We make sure we don't call pipeline with user_config
            del args['user_config']
            return self.pipeline(
                **{k: v for k, v in args.items() if k not in extra_args},  # Exclude extra_args
                **extra_args  # Include extra_args
            ).images
        else:
            # We make sure we don't call pipeline with unwanted args
            for unwanted_arg in ['prompt_embeds', 'negative_prompt_embeds', 'pooled_prompt_embeds', 'negative_pooled_prompt_embeds']:
                if unwanted_arg in args:
                    del args[unwanted_arg]
            return self.pipeline(
                **{k: v for k, v in args.items() if k not in extra_args},  # Exclude extra_args
                **extra_args  # Include extra_args
            ).images