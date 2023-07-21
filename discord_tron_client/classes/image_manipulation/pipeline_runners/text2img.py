from discord_tron_client.classes.image_manipulation.pipeline_runners.base_runner import BasePipelineRunner

class Text2ImgPipelineRunner(BasePipelineRunner):
    def __init__(self, pipeline):
        self.pipeline = pipeline
    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        user_config: dict,
        prompt_embeds,
        negative_prompt_embeds,
        width,
        height,
        num_inference_steps,
        denoising_end,
        guidance_scale,
        guidance_rescale,
        num_images_per_prompt,
        output_type,
        generator,
        pooled_prompt_embeds = None,
        negative_pooled_prompt_embeds = None,
    ):
        if user_config.get("prompt_weighting", True):
            return self.pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                num_inference_steps=int(float(num_inference_steps)),
                guidance_scale=float(guidance_scale),
                guidance_rescale=float(guidance_rescale),
                denoising_end=denoising_end,
                output_type=output_type,
                generator=generator,
            ).images
        else:
            return self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                num_inference_steps=int(float(num_inference_steps)),
                guidance_rescale=int(guidance_rescale),
                guidance_scale=float(guidance_scale),
                denoising_end=denoising_end,
                output_type=output_type,
                generator=generator
            ).images
