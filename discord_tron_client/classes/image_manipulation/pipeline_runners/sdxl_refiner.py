from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)


class SdxlRefinerPipelineRunner(BasePipelineRunner):
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
        guidance_scale,
        guidance_rescale,
        num_images_per_prompt,
        output_type,
        aesthetic_score,
        negative_aesthetic_score,
        generator,
        image = None,
        denoising_start = None,
        denoising_end = None,
    ):
        if user_config.get("refiner_prompt_weighting", False):
            # Currently, it seems like the refiner's prompt weighting is broken.
            # We are disabling it by default.
            num_images_per_prompt = 1 # SDXL, when using prompt embeds, only generates 1 image per prompt.
            return self.pipeline(
                image=image,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                num_inference_steps=int(float(num_inference_steps)),
                denoising_start=denoising_start,
                denoising_end=denoising_end,
                guidance_rescale=int(guidance_rescale),
                guidance_scale=float(guidance_scale),
                output_type=output_type,
                generator=generator,
            ).images
        else:
            return self.pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
                num_inference_steps=int(float(num_inference_steps)),
                denoising_start=denoising_start,
                denoising_end=denoising_end,
                guidance_rescale=int(guidance_rescale),
                guidance_scale=float(guidance_scale),
                output_type=output_type,
                generator=generator,
            ).images
