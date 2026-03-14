"""
FlashInfer attention wrapper using the .plan() / .run() API.

Wraps BatchPrefillWithPagedKVCacheWrapper and BatchDecodeWithPagedKVCacheWrapper
for prefill, decode, and tree attention modes.

Adapted from subspec_v2/specdecodes/models/utils/flashinfer/be_attention_wrapper.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import flashinfer

from .cache_manager import KvCacheBatchPosition

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

FLASH_INFER_SUPPORTED_DIMS = [64, 128, 256]


class POS_ENCODING_MODE(Enum):
    ROPE_LLAMA = "ROPE_LLAMA"
    ALIBI = "ALIBI"
    NONE = "NONE"


@dataclass(frozen=True)
class AttentionRotaryParams:
    causal: bool = True
    pos_encoding_mode: POS_ENCODING_MODE = POS_ENCODING_MODE.ROPE_LLAMA
    rope_scale: float = 1.0
    rope_theta: float = 1.0e4


def find_padded_head_dim(head_dim: int) -> int:
    for dim in FLASH_INFER_SUPPORTED_DIMS:
        if head_dim <= dim:
            return dim
    raise ValueError(f"Head dimension {head_dim} too large for FlashInfer (max {FLASH_INFER_SUPPORTED_DIMS[-1]})")


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class BeFlashinferWrapper:
    """
    FlashInfer paged attention wrapper.

    Supports prefill, decode, and tree (custom mask) modes.
    """

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        page_len: int,
        device: Optional[torch.device] = None,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self._head_padded_dim = find_padded_head_dim(self.head_dim)
        self.page_len = page_len

        self.group_size = self.num_attention_heads // self.num_key_value_heads
        _device = device if device is not None else torch.cuda.current_device()
        _workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.int8, device=_device
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer, kv_layout="NHD",
        )
        _use_tensor_cores = self.group_size in [7, 16]
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=_use_tensor_cores,
        )
        self._decode_output_buf = None

    # ------------------------------------------------------------------
    # CUDA graph initialization
    # ------------------------------------------------------------------

    def init_cuda_graph_decode(self, K: int, max_num_pages: int, device: torch.device):
        """Reinitialize decode_wrapper with use_cuda_graph=True."""
        _workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
        _use_tensor_cores = self.group_size in [7, 16]
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=_use_tensor_cores,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=torch.zeros(K + 1, dtype=torch.int32, device=device),
            paged_kv_indices_buffer=torch.zeros(max_num_pages, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buffer=torch.zeros(K, dtype=torch.int32, device=device),
        )

    # ------------------------------------------------------------------
    # prepareAttention — calls .plan() on the appropriate wrapper
    # ------------------------------------------------------------------

    def prepareAttention(
        self,
        mode: str,
        batch_position: KvCacheBatchPosition,
        page_len: int,
        pos_encoding_mode,
        dtype: torch.dtype,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Plan attention for the next forward pass.

        Args:
            mode: "prefill", "decode", or "tree"
            batch_position: KvCacheBatchPosition describing the batch
            dtype: query data type (activation dtype)
            attention_mask: custom attention mask (only for "tree" mode)
        """
        if mode == "tree" and attention_mask is not None:
            self.prefill_wrapper.plan(
                qo_indptr=batch_position.seq_indptr,
                paged_kv_indptr=batch_position.kv_page_indptr,
                paged_kv_indices=batch_position.kv_page_indices,
                paged_kv_last_page_len=batch_position.kv_last_page_len,
                num_qo_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim_qk=self._head_padded_dim,
                page_size=page_len,
                custom_mask=attention_mask,
                causal=False,
                q_data_type=dtype,
            )
        elif mode in ("tree", "prefill"):
            self.prefill_wrapper.plan(
                qo_indptr=batch_position.seq_indptr,
                paged_kv_indptr=batch_position.kv_page_indptr,
                paged_kv_indices=batch_position.kv_page_indices,
                paged_kv_last_page_len=batch_position.kv_last_page_len,
                num_qo_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim_qk=self._head_padded_dim,
                page_size=page_len,
                causal=True,
                q_data_type=dtype,
            )
        elif mode == "decode":
            self.decode_wrapper.plan(
                indptr=batch_position.kv_page_indptr,
                indices=batch_position.kv_page_indices,
                last_page_len=batch_position.kv_last_page_len,
                num_qo_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self._head_padded_dim,
                page_size=page_len,
                q_data_type=dtype,
            )
        else:
            raise ValueError(f"Invalid attention mode: {mode}")

    # ------------------------------------------------------------------
    # Reshape / pad / unpad helpers
    # ------------------------------------------------------------------

    def reshape_qkv_for_attention(self, q, k, v, batchPosition: KvCacheBatchPosition):
        """Reshape Q/K/V from [batch, seq, heads, dim] to [tokens, heads, dim]."""
        return (
            q.view(-1, self.num_attention_heads, self.head_dim),
            k.view(-1, self.num_key_value_heads, self.head_dim),
            v.view(-1, self.num_key_value_heads, self.head_dim),
        )

    def _unpad_attention(self, attn_output):
        if self._head_padded_dim > self.head_dim:
            return attn_output[:, :, :self.head_dim].reshape(-1, self.hidden_size)
        return attn_output.view(-1, self.hidden_size)

    def _pad_qkv(self, q, k, v):
        if self._head_padded_dim > self.head_dim:
            pad = self._head_padded_dim - self.head_dim
            q = torch.nn.functional.pad(q, (0, pad))
            k = torch.nn.functional.pad(k, (0, pad))
            v = torch.nn.functional.pad(v, (0, pad))
        return q, k, v

    # ------------------------------------------------------------------
    # KV cache append
    # ------------------------------------------------------------------

    def append_kv_cache(self, q, k, v, batch_position, paged_kv_cache, page_len):
        """Append K,V to the paged cache at the correct positions."""
        flashinfer.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=batch_position.batch_indices,
            paged_kv_cache=paged_kv_cache,
            kv_indices=batch_position.kv_page_indices,
            positions=batch_position.positions,
            kv_indptr=batch_position.kv_page_indptr,
            kv_last_page_len=batch_position.kv_last_page_len,
        )

    # ------------------------------------------------------------------
    # computeAttention — main entry point
    # ------------------------------------------------------------------

    def computeAttention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        mode: str,
        batchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
        layer_idx: int,
    ):
        """
        Compute attention: pad Q/K/V, append to cache, run attention kernel.

        Args:
            cacheData: single layer's cache [max_pages, 2, page_len, num_heads, head_dim]
            mode: "prefill", "decode", or "tree"
        """
        q, k, v = self._pad_qkv(q, k, v)
        if mode in ("prefill", "tree"):
            attn_output = self._batchPrefill(q, k, v, cacheData, batchPosition, rotaryParams)
        elif mode == "decode":
            attn_output = self._batchDecode(q, k, v, cacheData, batchPosition, rotaryParams)
        else:
            raise ValueError(f"Invalid attention mode: {mode}")
        return self._unpad_attention(attn_output)

    def _batchPrefill(self, q, k, v, cacheData, batchPosition, rotaryParams):
        self.append_kv_cache(q, k, v, batchPosition, cacheData, self.page_len)
        return self.prefill_wrapper.run(q, cacheData)

    def _batchDecode(self, q, k, v, cacheData, batchPosition, rotaryParams):
        self.append_kv_cache(q, k, v, batchPosition, cacheData, self.page_len)
        buf = self._decode_output_buf
        if buf is None or buf.shape[0] < q.shape[0]:
            buf = torch.empty_like(q)
            self._decode_output_buf = buf
        out = buf[:q.shape[0]]
        self.decode_wrapper.run(q, cacheData, out=out)
        return out
