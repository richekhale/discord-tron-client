import logging, sys, torch, tqdm, traceback, time
from discord_tron_client.classes.image_manipulation.resolution import ResolutionManager


class PipelineRunner:
    def __init__(self, model_manager, pipeline_manager, config):
        self.model_manager = model_manager
        self.pipeline_manager = pipeline_manager
        self.config = config

    def _prepare_pipe(self, model_id, use_attention_scaling):
        logging.info("Retrieving pipe for model " + str(model_id))
        pipe = self.pipeline_manager.get_pipe(model_id, use_attention_scaling)
        logging.info("Copied pipe to the local context")
        return pipe

    def _generate_image_with_pipe(self, pipe, prompt, side_x, side_y, steps, negative_prompt):
        with torch.no_grad():
            with tqdm(total=steps, ncols=100, file=sys.stderr) as pbar:
                image = pipe(
                    prompt=prompt,
                    height=side_y,
                    width=side_x,
                    num_inference_steps=int(float(steps)),
                    negative_prompt=negative_prompt,
                ).images[0]
        return image

    def generate_image( self, prompt, model_id, resolution, negative_prompt, steps, positive_prompt, user_config, discord_msg, websocket):
        logging.info("Initializing image generation pipeline...")
        tqdm_capture = tqdm.tqdm.write
        use_attention_scaling, steps = self.check_attention_scaling(resolution, steps)
        aspect_ratio, side_x, side_y = ResolutionManager.get_aspect_ratio_and_sides(self.config, resolution)

        pipe = self._prepare_pipe(model_id, use_attention_scaling)
        logging.info("REDIRECTING THE PRECIOUS, STDOUT... SORRY IF THAT UPSETS YOU")

        original_stderr = sys.stderr
        sys.stderr = tqdm_capture

        entire_prompt = self.combine_prompts(prompt, positive_prompt)

        for attempt in range(1, 6):
            try:
                logging.info(f"Attempt {attempt}: Generating image...")
                image = self._generate_image_with_pipe(pipe, entire_prompt, side_x, side_y, steps, negative_prompt)
                logging.info("Image generation successful!")

                scaling_target = ResolutionManager.nearest_scaled_resolution(resolution, user_config, self.config.get_max_resolution_by_aspect_ratio(aspect_ratio))
                if scaling_target != resolution:
                    logging.info("Rescaling image to nearest resolution...")
                    image = image.resize((scaling_target["width"], scaling_target["height"]))
                return image
            except Exception as e:
                logging.error(f"Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}")
                if attempt < 5:
                    time.sleep(5)
                else:
                    raise RuntimeError("Maximum retries reached, image generation failed")
            finally:
                sys.stderr = original_stderr
    def check_attention_scaling(self, resolution, steps):
        is_attn_enabled = self.config.get_attention_scaling_status()
        use_attention_scaling = False
        if resolution is not None and is_attn_enabled:
            scaling_factor = self.get_scaling_factor(
                resolution["width"], resolution["height"], self.resolutions
            )
            logging.info(
                f"Scaling factor for {resolution['width']}x{resolution['height']}: {scaling_factor}"
            )
            if scaling_factor < 50:
                logging.info(
                    "Resolution "
                    + str(resolution["width"])
                    + "x"
                    + str(resolution["height"])
                    + " has a pixel count greater than threshold. Using attention scaling expects to take 30 seconds."
                )
                use_attention_scaling = True
                if steps > scaling_factor:
                    steps = scaling_factor
        return use_attention_scaling, steps

    def combine_prompts(self, prompt, positive_prompt):
        entire_prompt = prompt
        if positive_prompt is not None:
            entire_prompt = str(prompt) + " , " + str(positive_prompt)
        return entire_prompt
