"""
HQQ quantization helpers.

Provides functions to generate per-layer quantization configs and
apply HQQ quantization with GemLite backend.

Adapted from subspec_v2/specdecodes/helpers/quantizers/hqq/__init__.py
and subspec_v2/specdecodes/helpers/recipes/quant/hqq_4bit.py
"""

from __future__ import annotations

import logging

import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
from hqq.models.hf.base import AutoHQQHFModel
from hqq.utils.patching import prepare_for_inference


def make_quant_config(
    model: torch.nn.Module,
    nbits: int = 4,
    group_size: int = 64,
    axis: int = 1,
) -> dict:
    """
    Generate per-layer HQQ quantization config for all attention + MLP layers.

    Args:
        model: HuggingFace model (must have model.model.layers)
        nbits: Number of bits (1, 2, 3, 4, 8)
        group_size: Quantization group size
        axis: Grouping axis (0=better quality, 1=faster inference)

    Returns:
        Dict mapping layer name → BaseQuantizeConfig
    """
    quant_config = {}
    config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=axis)

    # HQQ uses short "linear_tags" (e.g. "self_attn.q_proj") that apply
    # the same config to all layers. Each tag needs its own config dict
    # because HQQLinear.__init__ mutates (pops from) the config.
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        quant_config[f"self_attn.{proj}"] = BaseQuantizeConfig(
            nbits=nbits, group_size=group_size, axis=axis
        )
    for proj in ("gate_proj", "up_proj", "down_proj"):
        quant_config[f"mlp.{proj}"] = BaseQuantizeConfig(
            nbits=nbits, group_size=group_size, axis=axis
        )

    return quant_config


def quantize_model(
    model: torch.nn.Module,
    quant_config: dict,
    compute_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> None:
    """
    Apply HQQ quantization to model with GemLite backend.

    Args:
        model: The model to quantize (modified in-place)
        quant_config: Per-layer quantization config from make_quant_config()
        compute_dtype: Activation dtype
        device: Target device
    """
    logging.info(f"Quantizing model with HQQ (config has {len(quant_config)} layers)")

    AutoHQQHFModel.quantize_model(
        model, quant_config, compute_dtype=compute_dtype, device=device
    )
    HQQLinear.set_backend(HQQBackend.PYTORCH)

    try:
        prepare_for_inference(model, backend="gemlite")
        logging.info("HQQ model patched with GemLite backend")
    except Exception as e:
        logging.warning(f"GemLite patching failed, using PyTorch backend: {e}")
