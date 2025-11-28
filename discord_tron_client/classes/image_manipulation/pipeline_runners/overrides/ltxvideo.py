import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
    USE_PEFT_BACKEND,
)

logger = logging.get_logger(__name__)


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> torch.Tensor:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if (
            attention_kwargs is not None
            and attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    image_rotary_emb = self.rope(
        hidden_states, num_frames, height, width, rope_interpolation_scale
    )

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (
            1 - encoder_attention_mask.to(hidden_states.dtype)
        ) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    batch_size = hidden_states.size(0)
    hidden_states = self.proj_in(hidden_states)

    temb, embedded_timestep = self.time_embed(
        timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )

    temb = temb.view(batch_size, -1, temb.size(-1))
    embedded_timestep = embedded_timestep.view(
        batch_size, -1, embedded_timestep.size(-1)
    )

    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(
        batch_size, -1, hidden_states.size(-1)
    )

    if self.enable_teacache:
        print(f"enabled teacache")
        inp = hidden_states.clone()
        temb_ = temb.clone()
        inp = self.transformer_blocks[0].norm1(inp)
        num_ada_params = self.transformer_blocks[0].scale_shift_table.shape[0]
        ada_values = self.transformer_blocks[0].scale_shift_table[
            None, None
        ] + temb_.reshape(batch_size, temb_.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            ada_values.unbind(dim=2)
        )
        modulated_inp = inp * (1 + scale_msa) + shift_msa
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [
                2.14700694e01,
                -1.28016453e01,
                2.31279151e00,
                7.92487521e-01,
                9.69274326e-03,
            ]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            print(f"should calc: {should_calc}")
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                    )

            scale_shift_values = (
                self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            )
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )

        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


import contextlib
import types

# A unique sentinel to mark attributes that did not exist originally.
_SENTINEL = object()


@contextlib.contextmanager
def ltx_teacache_monkeypatch(
    pipeline, num_inference_steps, rel_l1_thresh=0.6, disable=False
):
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
