from discord_tron_client.classes.image_manipulation.pipeline_runners.base_runner import BasePipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.text2img import Text2ImgPipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.img2img import Img2ImgPipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.sdxl_base import SdxlBasePipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.sdxl_refiner import SdxlRefinerPipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.kandinsky_2_2 import KandinskyTwoTwoPipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.deep_floyd import DeepFloydPipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.sd3_runner import SD3PipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.aura import AuraPipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.pixart import PixArtPipelineRunner
from discord_tron_client.classes.image_manipulation.pipeline_runners.flux import FluxPipelineRunner

runner_map = {
    "text2img": Text2ImgPipelineRunner,
    "img2img": Img2ImgPipelineRunner,
    "sdxl_base": SdxlBasePipelineRunner,
    "sdxl_refiner": SdxlRefinerPipelineRunner,
    "kandinsky_2.2": KandinskyTwoTwoPipelineRunner,
    "deep_floyd": DeepFloydPipelineRunner,
    "sd3": SD3PipelineRunner,
    "pixart": PixArtPipelineRunner,
    "aura": AuraPipelineRunner,
    "flux": FluxPipelineRunner,
}
