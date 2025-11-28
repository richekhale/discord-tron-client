import logging
from typing import Any
import numpy as np
import torch

from discord_tron_client.classes.image_manipulation.pipeline_runners import (
    BasePipelineRunner,
)


class _DirectPipelineRunner(BasePipelineRunner):
    """
    Minimal runner that forwards arguments to the underlying pipeline while normalizing
    types and dropping user_config. Used for models that don't need special handling.
    """

    def _normalize_args(self, args: dict) -> dict:
        numeric_keys = [
            "num_inference_steps",
            "prior_num_inference_steps",
            "num_frames",
            "num_videos_per_prompt",
            "height",
            "width",
        ]
        float_keys = ["guidance_scale", "prior_guidance_scale", "decoder_guidance_scale"]
        for key in numeric_keys:
            if key in args and args[key] is not None:
                try:
                    args[key] = int(args[key])
                except Exception:
                    logging.debug(f"Could not cast {key}={args[key]} to int")
        for key in float_keys:
            if key in args and args[key] is not None:
                try:
                    args[key] = float(args[key])
                except Exception:
                    logging.debug(f"Could not cast {key}={args[key]} to float")
        return args

    def _run_pipeline(self, args: dict):
        result = self.pipeline(**args)
        if hasattr(result, "images"):
            return result.images
        if hasattr(result, "videos"):
            return result.videos
        if hasattr(result, "audios"):
            return result.audios
        if hasattr(result, "paths"):
            return result.paths
        return result

    def __call__(self, **args: Any):
        prompt_value = args.get("prompt")
        if prompt_value is not None:
            args["prompt"], prompt_parameters = self._extract_parameters(prompt_value)
            args.update(prompt_parameters)
        args.pop("user_config", None)
        args = self._normalize_args(args)
        logging.debug(f"Args for {_safe_name(self)}: {args}")
        return self._run_pipeline(args)


def _safe_name(obj: Any) -> str:
    return obj.__class__.__name__


class StableCascadePipelineRunner(_DirectPipelineRunner):
    pass


class Flux2PipelineRunner(_DirectPipelineRunner):
    pass


class Kandinsky5ImagePipelineRunner(_DirectPipelineRunner):
    pass


class Kandinsky5VideoPipelineRunner(_DirectPipelineRunner):
    def _run_pipeline(self, args: dict):
        result = self.pipeline(**args)
        if hasattr(result, "videos"):
            return result.videos
        return super()._run_pipeline(args)


class CosmosPipelineRunner(_DirectPipelineRunner):
    pass


class WanPipelineRunner(_DirectPipelineRunner):
    pass


class Lumina2PipelineRunner(_DirectPipelineRunner):
    pass


class OmniGenPipelineRunner(_DirectPipelineRunner):
    pass


class ACEStepPipelineRunner(_DirectPipelineRunner):
    sample_rate = 48000

    def __call__(self, **args: Any):
        prompt_value = args.get("prompt")
        if prompt_value is not None:
            args["prompt"], prompt_parameters = self._extract_parameters(prompt_value)
            args.update(prompt_parameters)
        args.pop("user_config", None)
        args = self._normalize_args(args)

        result = self.pipeline(**args)
        audios = getattr(result, "audios", None)
        if audios is None:
            audios = result

        audio_arr = None
        if isinstance(audios, (list, tuple)) and len(audios) > 0:
            first = audios[0]
            if isinstance(first, torch.Tensor):
                audio_arr = first.detach().cpu().numpy()
            elif isinstance(first, np.ndarray):
                audio_arr = first
            else:
                audio_arr = np.asarray(first)

        paths = getattr(result, "paths", None)
        params = getattr(result, "params", None)

        payload = {
            "audio": audio_arr,
            "sample_rate": self.sample_rate,
            "paths": paths,
            "params": params,
        }
        return payload
