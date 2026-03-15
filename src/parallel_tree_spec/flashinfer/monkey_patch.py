"""
Monkey-patch HuggingFace Llama models to use FlashInfer paged attention.

Replaces LlamaAttention and optionally LlamaRMSNorm with FlashInfer-backed versions.

Adapted from subspec_v2/specdecodes/models/utils/flashinfer/monkey_patch.py
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention

from .attention import FiLlamaAttention
from .modeling_llama import llama_causal_lm_forward, llama_model_forward

try:
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm

    class FiLlamaRMSNorm(nn.Module):
        """FlashInfer-backed RMS normalization."""

        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def extra_repr(self):
            return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

        def forward(
            self,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            bsz, seq_len, hidden_size = hidden_states.size()
            if residual is not None:
                fused_add_rmsnorm(hidden_states, residual, self.weight.data, self.variance_epsilon)
                return hidden_states, residual
            hidden_states = rmsnorm(
                hidden_states.view(bsz * seq_len, hidden_size),
                self.weight,
                eps=self.variance_epsilon,
            )
            return hidden_states.view(bsz, seq_len, hidden_size)

    _HAS_FI_RMSNORM = True
except ImportError:
    _HAS_FI_RMSNORM = False


def _bind_method_to_module(module: nn.Module, method_name: str, new_method: Callable):
    """Bind a new method to a module instance so self is passed correctly."""
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_rms_norm_module(module: nn.Module, eps: float = 1e-6):
    """Replace RMSNorm forward with FlashInfer fused version."""
    module.variance_epsilon = (
        getattr(module, "variance_epsilon", None)
        or getattr(module, "eps", None)
        or eps
    )
    _bind_method_to_module(module, "forward", FiLlamaRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", FiLlamaRMSNorm.extra_repr)


def _patch_attention_module(module: nn.Module):
    """Replace attention forward with FlashInfer paged attention."""
    if isinstance(module, FiLlamaAttention):
        return  # already patched
    if isinstance(module, LlamaAttention):
        _bind_method_to_module(module, "forward", FiLlamaAttention.forward)
    else:
        raise ValueError(f"Unsupported attention module type: {type(module)}")


def apply_flashinfer_kernel_to_llama(
    model: PreTrainedModel,
    attention: bool = True,
    rms_norm: bool = True,
) -> None:
    """
    Monkey-patch a HuggingFace Llama model to use FlashInfer kernels.

    Args:
        model: The HF model to patch (must be loaded already)
        attention: Whether to patch attention layers
        rms_norm: Whether to patch RMSNorm layers
    """
    from transformers.models.llama import modeling_llama

    if rms_norm and _HAS_FI_RMSNORM:
        modeling_llama.LlamaRMSNorm = FiLlamaRMSNorm
    if attention:
        modeling_llama.LlamaAttention = FiLlamaAttention

    # Patch the model instance's layers
    if hasattr(model, "base_model_prefix"):
        base_model = getattr(model, model.base_model_prefix, model)
    else:
        base_model = getattr(model, "model", model).model

    if rms_norm and _HAS_FI_RMSNORM:
        _patch_rms_norm_module(base_model.norm)

    # Patch out causal mask computation (FlashInfer handles masking internally)
    if attention:
        _bind_method_to_module(base_model, "forward", llama_model_forward)
        if base_model is not model:
            # model is a CausalLM wrapper around base_model
            _bind_method_to_module(model, "forward", llama_causal_lm_forward)

    for decoder_layer in base_model.layers:
        if rms_norm and _HAS_FI_RMSNORM:
            _patch_rms_norm_module(decoder_layer.input_layernorm)
            _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
        if attention:
            _patch_attention_module(decoder_layer.self_attn)
