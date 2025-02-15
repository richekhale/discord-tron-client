import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, logging, scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND

logger = logging.get_logger(__name__)

def teacache_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    pooled_projections: torch.FloatTensor = None,
    timestep: torch.LongTensor = None,
    block_controlnet_hidden_states: Optional[List[torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    skip_layers: Optional[List[int]] = None,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    Replacement forward pass for SD3Transformer2DModel that includes a TeaCache-like mechanism.

    If `self.enable_teacache` is set, we measure how much the input changes from the last iteration
    and decide whether to skip internal computation. If we skip, we just add the previous residual.

    IMPORTANT: We now check shape consistency, because batch size can differ (e.g. during CFG).
               If the current hidden_states shape doesn't match the stored previous_residual,
               we can't skip.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and "scale" in joint_attention_kwargs:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    original_hidden_states = hidden_states

    if getattr(self, "enable_teacache", False):
        # Compute a “should_calc” bool, or skip.
        # --------------------------------------------------
        # (1) If it’s the first or last step, we always compute
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            old = self.previous_modulated_input
            if old is not None and old.abs().mean() > 1e-8:
                rel_diff = (hidden_states - old).abs().mean() / old.abs().mean()
            else:
                rel_diff = torch.tensor(float("inf"), device=hidden_states.device)

            self.accumulated_rel_l1_distance += rel_diff.item()
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0.0

        self.previous_modulated_input = hidden_states
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0  # reset for next call

        # (2) If we do NOT want to compute
        if not should_calc:
            # only skip if previous_residual is not None AND shapes match
            if self.previous_residual is not None:
                # shape check
                if hidden_states.shape == self.previous_residual.shape:
                    output = hidden_states + self.previous_residual

                    if USE_PEFT_BACKEND:
                        unscale_lora_layers(self, lora_scale)

                    if not return_dict:
                        return (output,)
                    return Transformer2DModelOutput(sample=output)
                else:
                    # shape mismatch => do not skip
                    pass

    # ---------------
    # Original forward logic
    # ---------------
    height, width = hidden_states.shape[-2:]
    hidden_states = self.pos_embed(hidden_states)
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    for index_block, block in enumerate(self.transformer_blocks):
        # skip_layers logic
        if skip_layers is not None and index_block in skip_layers:
            # apply controlnet skip anyway
            if block_controlnet_hidden_states is not None and not block.context_pre_only:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states += block_controlnet_hidden_states[index_block // interval_control]
            continue

        # gradient checkpoint logic
        if (
            self.training
            and self.gradient_checkpointing
            and (
                self.gradient_checkpointing_interval is None
                or index_block % self.gradient_checkpointing_interval == 0
            )
        ):
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    return module(*inputs)
                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                **ckpt_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        if block_controlnet_hidden_states is not None and not block.context_pre_only:
            interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
            hidden_states += block_controlnet_hidden_states[index_block // interval_control]

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # unpatchify
    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(
            hidden_states.shape[0],
            height,
            width,
            patch_size,
            patch_size,
            self.out_channels,
        )
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(
            hidden_states.shape[0],
            self.out_channels,
            height * patch_size,
            width * patch_size,
        )
    )

    # ---------------
    # If TeaCache is on, store new residual if shapes match
    # ---------------
    if getattr(self, "enable_teacache", False):
        if output.shape == original_hidden_states.shape:
            self.previous_residual = output - original_hidden_states
        else:
            self.previous_residual = None

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

import contextlib
import types

# A unique sentinel to mark attributes that did not exist originally.
_SENTINEL = object()

@contextlib.contextmanager
def sd3_teacache_monkeypatch(pipeline, num_inference_steps, rel_l1_thresh=0.6, disable=False):
    """
    Temporarily monkeypatches the transformer's instance in the given pipeline.

    If disable is False (the default), this context manager:
      - Binds and overrides the instance's forward method with teacache_forward.
      - Sets several attributes on the transformer instance:
            enable_teacache, cnt, num_steps, rel_l1_thresh,
            accumulated_rel_l1_distance, previous_modulated_input, previous_residual.
    If disable is True, the forward method is left unmodified and enable_teacache is set to False.

    Parameters:
      pipeline: A DiffusionPipeline instance with a `.transformer` attribute.
      num_inference_steps: The number of inference steps to set.
      rel_l1_thresh: The relative L1 threshold (default is 0.6).
      disable: If True, disables teaCache (forward remains unchanged and enable_teacache is False).
    """
    transformer = pipeline.transformer

    # Save the original forward method from the instance (if it exists).
    orig_forward = transformer.__dict__.get("forward", _SENTINEL)

    # Prepare the attributes to patch; note that enable_teacache is based on disable.
    attrs_to_patch = {
        "enable_teacache": not disable,
        "cnt": 0,
        "num_steps": num_inference_steps,
        "rel_l1_thresh": rel_l1_thresh,
        "accumulated_rel_l1_distance": 0,
        "previous_modulated_input": None,
        "previous_residual": None,
    }

    # Save the original attribute values (if they exist on the instance).
    original_attrs = {}
    for attr in attrs_to_patch:
        if attr in transformer.__dict__:
            original_attrs[attr] = transformer.__dict__[attr]
        else:
            original_attrs[attr] = _SENTINEL

    try:
        if not disable:
            # Bind teacache_forward to the transformer instance so that self is passed correctly.
            transformer.forward = types.MethodType(teacache_forward, transformer)
        else:
            # If disabling, make sure we don't have an instance-specific forward override.
            if "forward" in transformer.__dict__:
                delattr(transformer, "forward")

        # Apply the patched attributes.
        for attr, value in attrs_to_patch.items():
            setattr(transformer, attr, value)

        # Yield control to the caller with the patch active.
        yield pipeline
    finally:
        # Restore the original forward method.
        if orig_forward is _SENTINEL:
            if "forward" in transformer.__dict__:
                delattr(transformer, "forward")
        else:
            transformer.forward = orig_forward

        # Restore all the other original attributes.
        for attr, orig_value in original_attrs.items():
            if orig_value is _SENTINEL:
                if attr in transformer.__dict__:
                    delattr(transformer, attr)
            else:
                setattr(transformer, attr, orig_value)
