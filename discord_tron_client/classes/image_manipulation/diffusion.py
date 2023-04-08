from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline

class DiffusionPipelineManager:
    def __init__(self):
        self.pipelines = {}

    def get_pipe(self, pipe_id, use_attn_scaling: bool = False):
        if pipe_id not in self.pipelines:
            self.pipelines[pipe_id] = StableDiffusionPipeline.from_pretrained(pipe_id)
        return self.pipelines[pipe_id]