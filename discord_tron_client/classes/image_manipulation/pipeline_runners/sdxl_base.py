from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)


class SdxlBasePipelineRunner(BasePipelineRunner):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(
        self,
        prompt,
        negative_prompt,
        user_config,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        width,
        height,
        num_inference_steps,
        denoising_end,
        guidance_scale,
        guidance_rescale,
        num_images_per_prompt,
        output_type,
        generator,
    ):
        if user_config.get("prompt_weighting", True):
            num_images_per_prompt = 1 # SDXL, when using prompt embeds, only generates 1 image per prompt.
            return self.pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                num_inference_steps=int(float(num_inference_steps)),
                denoising_end=denoising_end,
                guidance_rescale=int(guidance_rescale),
                guidance_scale=float(guidance_scale),
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
                denoising_end=denoising_end,
                guidance_rescale=int(guidance_rescale),
                guidance_scale=float(guidance_scale),
                output_type=output_type,
                generator=generator,
            ).images
