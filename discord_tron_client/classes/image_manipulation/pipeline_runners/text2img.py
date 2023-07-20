from discord_tron_client.classes.image_manipulation.pipeline_runners.base_runner import BasePipelineRunner

class Text2ImgPipelineRunner(BasePipelineRunner):
    def __init__(self, pipeline):
        self.pipeline = pipeline
