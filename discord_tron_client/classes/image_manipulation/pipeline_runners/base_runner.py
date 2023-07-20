from PIL import Image
from diffusers import DiffusionPipeline

class BasePipelineRunner:
    def __init__(self, pipeline: DiffusionPipeline):
        self.pipeline = pipeline

    def run(self) -> Image:
        raise NotImplementedError