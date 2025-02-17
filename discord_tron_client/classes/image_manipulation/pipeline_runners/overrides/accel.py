import contextlib, logging
from DeepCache import DeepCacheSDHelper
from discord_tron_client.classes.image_manipulation.pipeline_runners.overrides.flux import (
    flux_teacache_monkeypatch,
)
from discord_tron_client.classes.image_manipulation.pipeline_runners.overrides.sd3 import (
    sd3_teacache_monkeypatch,
)

sage_mechanisms = {}
try:
    from sageattention import (
        sageattn_qk_int8_pv_fp16_triton,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        sageattn_varlen,
        sageattn,
    )

    sage_mechanisms = {
        "sageattn_qk_int8_pv_fp16_triton": sageattn_qk_int8_pv_fp16_triton,
        "sageattn_qk_int8_pv_fp16_cuda": sageattn_qk_int8_pv_fp16_cuda,
        "sageattn_qk_int8_pv_fp8_cuda": sageattn_qk_int8_pv_fp8_cuda,
        "sageattn_qk_int8_pv_fp8_cuda_sm90": sageattn_qk_int8_pv_fp8_cuda_sm90,
        "sageattn_varlen": sageattn_varlen,
        "sageattn": sageattn,
    }
except ImportError:
    pass

def enable_sageattention(sageattention_mechanism: str = "sageattn"):
    from torch.nn import functional as F

    original_attention = F.scaled_dot_product_attention
    def sdpa_hijack_flash(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        try:
            return sage_mechanisms[sageattention_mechanism](query, key, value, attn_mask, dropout_p, is_causal, scale)
        except Exception:
            hidden_states = original_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
        return hidden_states
    logging.info(f"Enabled SageAttention mechanism: {sageattention_mechanism}")
    F.scaled_dot_product_attention = sdpa_hijack_flash

    return original_attention


def disable_sageattention(original_attention):
    from torch.nn import functional as F

    F.scaled_dot_product_attention = original_attention


@contextlib.contextmanager
def optimize_pipeline(
    pipeline,
    # TeaCache toggles
    enable_teacache: bool = False,
    teacache_num_inference_steps: int = 20,
    teacache_rel_l1_thresh: float = 0.6,
    # DeepCache toggles
    enable_deepcache: bool = False,
    deepcache_cache_interval: int = 3,
    deepcache_cache_branch_id: int = 0,
    deepcache_skip_mode: str = "uniform",
    enable_sageattn: bool = False,
    sageattention_mechanism: str = "sageattn",
):
    """
    A unified context manager that enables TeaCache on `pipeline.transformer` (if present)
    and DeepCache on `pipeline.unet` (if present).

    Arguments:
        pipeline: The pipeline object (e.g. FluxPipeline or StableDiffusionPipeline).
        enable_teacache: If True, will apply TeaCache monkeypatch to pipeline.transformer.
        teacache_num_inference_steps: The number of inference steps to pass to TeaCache.
        teacache_rel_l1_thresh: The relative L1 threshold for TeaCache.
        enable_deepcache: If True, will create (if missing) and enable DeepCache on pipeline.unet.
        deepcache_cache_interval: Interval at which the unet forward pass is cached.
        deepcache_cache_branch_id: Branch ID for DeepCache.
        deepcache_skip_mode: Strategy for skipping unet blocks (e.g. "uniform").
        enable_sageattn: If True, will enable SageAttention mechanism.
        sageattention_mechanism: The SageAttention mechanism to use.
    """

    # --------------------------
    # 1. TeaCache Setup
    # --------------------------
    # If the pipeline has a `transformer` attribute and the user wants to enable TeaCache,
    # we wrap it with the teacache_monkeypatch context manager. Otherwise do a "no-op" context.
    # If no transformer, do a dummy context
    @contextlib.contextmanager
    def _nullctx():
        yield pipeline

    teacache_ctx = _nullctx()
    if (
        hasattr(pipeline, "transformer")
        and getattr(pipeline, "transformer") is not None
    ):
        if "flux" in str(type(pipeline.transformer)).casefold():
            logging.info("Optimizing Flux pipeline with TeaCache")
            teacache_ctx = flux_teacache_monkeypatch(
                pipeline,
                num_inference_steps=teacache_num_inference_steps,
                rel_l1_thresh=teacache_rel_l1_thresh,
                disable=(not enable_teacache),
            )
        elif "sd3" in str(type(pipeline.transformer)).casefold():
            logging.info("Optimizing SD3 pipeline with TeaCache")
            teacache_ctx = sd3_teacache_monkeypatch(
                pipeline,
                num_inference_steps=teacache_num_inference_steps,
                rel_l1_thresh=teacache_rel_l1_thresh,
                disable=(not enable_teacache),
            )
        logging.info(f"TeaCache enabled: {enable_teacache}, pipeline type: {type(pipeline.transformer)}")

    # --------------------------
    # 2. DeepCache Setup
    # --------------------------
    deepcache_active = False
    if enable_deepcache and hasattr(pipeline, "unet") and pipeline.unet is not None:
        # We only proceed if pipeline has unet
        # Make sure we have the `deepcache_helper` attribute
        if not hasattr(pipeline, "deepcache_helper"):
            try:
                helper = DeepCacheSDHelper(pipe=pipeline)
                # Set defaults
                helper.set_params(
                    cache_interval=deepcache_cache_interval,
                    cache_branch_id=deepcache_cache_branch_id,
                    skip_mode=deepcache_skip_mode,
                )
                setattr(pipeline, "deepcache_helper", helper)
            except Exception as e:
                print(f"[optimize_pipeline] Could not enable DeepCache: {e}")
                enable_deepcache = False

    # --------------------------
    # 3. Combine context managers
    # --------------------------
    with teacache_ctx:
        if enable_sageattn:
            original_attention = enable_sageattention(sageattention_mechanism)
        if enable_deepcache and hasattr(pipeline, "deepcache_helper"):
            pipeline.deepcache_helper.set_params(
                cache_interval=deepcache_cache_interval,
                cache_branch_id=deepcache_cache_branch_id,
                skip_mode=deepcache_skip_mode,
            )
            pipeline.deepcache_helper.enable()
            deepcache_active = True

        try:
            yield pipeline
        finally:
            # Cleanup / disable
            if deepcache_active:
                pipeline.deepcache_helper.disable()
            if enable_sageattn:
                disable_sageattention(original_attention)
