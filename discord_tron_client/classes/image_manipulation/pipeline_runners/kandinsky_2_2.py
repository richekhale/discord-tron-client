from discord_tron_client.classes.image_manipulation.pipeline_runners.base_runner import (
    BasePipelineRunner,
)
import logging

class KandinskyTwoTwoRunner(BasePipelineRunner):
    def __init__(self, decoder, pipeline_manager):
        self.decoder = decoder
        self.prior = None
        self.pipeline_manager = pipeline_manager

    def _setup_prior(self, user_config):
        prior_model = "kandinsky-community/kandinsky-2-2-prior"
        scheduler_config = {}  # This isn't really used anymore.
        if self.prior is not None:
            logging.info('Keeping existing Kandinsky 2.2 prior.')
            return
        self.prior = self.pipeline_manager.get_pipe(
            model_id=prior_model,
            user_config=user_config,
            scheduler_config=scheduler_config,
        )

    def __call__(self, **args):
        # Get user_config and delete it from args, it doesn't get passed to the pipeline
        user_config = args.get("user_config", None)
        del args["user_config"]
        self._setup_prior(user_config)

        # Obtain the embeddings from the prior:
        prompt = args.get('prompt', '')
        negative_prompt = args.get('negative_prompt', '')
        image_embeds, negative_image_embeds = self.prior(prompt, negative_prompt, guidance_scale=1.0).to_tuple()
        return self.decoder(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=args.get('height', 768),
            width=args.get('width', 768),
        ).images
