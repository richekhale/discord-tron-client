from typing import List, Union
import numpy as np
from PIL import Image
import imageio
import tempfile
from diffusers.utils.import_utils import BACKENDS_MAPPING, is_imageio_available, is_opencv_available
from diffusers.utils.export_utils import _legacy_export_to_video
import logging
import PIL

logger = logging.getLogger(__name__)

def export_to_video(
    video_frames: Union[List[np.ndarray], List[Image.Image]], output_video_path: str = None, fps: int = 10
) -> str:
    # TODO: Dhruv. Remove by Diffusers release 0.33.0
    # Added to prevent breaking existing code
    if not is_imageio_available():
        logger.warning(
            (
                "It is recommended to use `export_to_video` with `imageio` and `imageio-ffmpeg` as a backend. \n"
                "These libraries are not present in your environment. Attempting to use legacy OpenCV backend to export video. \n"
                "Support for the OpenCV backend will be deprecated in a future Diffusers version"
            )
        )
        return _legacy_export_to_video(video_frames, output_video_path, fps)

    if is_imageio_available():
        import imageio
    else:
        raise ImportError(BACKENDS_MAPPING["imageio"][1].format("export_to_video"))

    try:
        imageio.plugins.ffmpeg.get_exe()
    except AttributeError:
        raise AttributeError(
            (
                "Found an existing imageio backend in your environment. Attempting to export video with imageio. \n"
                "Unable to find a compatible ffmpeg installation in your environment to use with imageio. Please install via `pip install imageio-ffmpeg"
            )
        )

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(output_video_path, fps=fps, format='MP4') as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path
