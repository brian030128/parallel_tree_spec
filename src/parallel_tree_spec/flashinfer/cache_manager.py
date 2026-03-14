"""
Paged KV cache management for FlashInfer.

Provides:
- KvCachePool: GPU tensor pool of fixed-size pages
- RequestKvCache: Per-request page allocation/tracking
- KvCacheBatchPosition: Batch descriptor for FlashInfer kernels
- getKvCacheBatchPosition: Builds batch position from request caches

Adapted from subspec_v2/specdecodes/models/utils/flashinfer/cache_manager.py
"""

from __future__ import annotations

import math
import os
from typing import List

import torch
import flashinfer


class KvCacheBatchPosition:
    """Describes a batch of sequences for FlashInfer paged attention kernels."""

    def __init__(
        self,
        seq_indptr: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
    ):
        self.seq_indptr = seq_indptr            # [batch+1] cumulative token counts
        self.kv_page_indptr = kv_page_indptr    # [batch+1] cumulative page counts
        self.kv_page_indices = kv_page_indices  # [total_pages] physical page IDs
        self.kv_last_page_len = kv_last_page_len  # [batch] filled slots in last page
        self.batch_indices = batch_indices       # [total_tokens] batch item per token
        self.positions = positions               # [total_tokens] absolute positions

    def print_info(self):
        print(f"  seq_indptr:       {self.seq_indptr}")
        print(f"  kv_page_indptr:   {self.kv_page_indptr}")
        print(f"  kv_page_indices:  {self.kv_page_indices}")
        print(f"  kv_last_page_len: {self.kv_last_page_len}")
        print(f"  batch_indices:    {self.batch_indices}")
        print(f"  positions:        {self.positions}")


class KvCachePool:
    """
    Pool of paged KV cache on GPU.

    cache_data shape: [num_layers, max_pages, 2, page_len, num_heads, head_dim]
                      2 = keys (0) and values (1)
    """

    def __init__(
        self,
        max_pages: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.cache_data = torch.zeros(
            num_layers, max_pages, 2, page_len, num_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.num_layers = num_layers
        self.device = device
        self.max_pages = max_pages
        self.page_len = page_len
        self.num_heads = num_heads
        self.head_dims = head_dim
        self.dtype = dtype
        self.free_page_mask = torch.ones(max_pages, dtype=torch.bool, device="cpu")

    def reset(self):
        self.cache_data.zero_()
        self.free_page_mask.fill_(True)

    def num_free_pages(self):
        return self.free_page_mask.sum().item()

    def allocate(self, num_pages: int) -> List[int]:
        """Allocate num_pages free pages. Returns list of page indices."""
        free_page_indices = self.free_page_mask.nonzero()
        assert len(free_page_indices) >= num_pages, (
            f"Out of cache pages: asked {num_pages}, only {len(free_page_indices)} free"
        )
        allocated_indices = free_page_indices[:num_pages]
        self.free_page_mask[allocated_indices] = False
        return allocated_indices.squeeze(1).tolist()

    def deallocate(self, kv_page_indices: List[int]):
        """Return pages to the free pool."""
        self.free_page_mask[kv_page_indices] = True

    def reorder_cache_with_offset(
        self, beam_idx: torch.LongTensor, offset: int = 0, num_new_tokens: int = 0
    ):
        """
        Reorder cache for speculative decoding verification.
        Positions [:offset] remain unchanged. Tokens at beam_idx+offset are
        copied to positions [offset, offset+beam_size).
        """
        beam_idx = beam_idx.to(self.device)
        beam_size = beam_idx.size(0)

        old_indices = beam_idx + offset
        new_indices = torch.arange(offset, offset + beam_size, device=self.device, dtype=torch.long)

        page_len = self.page_len

        old_page_indices = old_indices // page_len
        old_token_indices = old_indices % page_len
        new_page_indices = new_indices // page_len
        new_token_indices = new_indices % page_len

        old_flat = old_page_indices * page_len + old_token_indices
        new_flat = new_page_indices * page_len + new_token_indices

        L, max_pages, _, page_len_, num_heads, head_dim = self.cache_data.shape

        # Separate K and V, flatten page+token dims, reorder, unflatten
        k_cat = self.cache_data[:, :, 0, :, :, :].clone().view(L, max_pages * page_len, num_heads, head_dim)
        v_cat = self.cache_data[:, :, 1, :, :, :].clone().view(L, max_pages * page_len, num_heads, head_dim)

        k_cat.index_copy_(1, new_flat, k_cat.index_select(1, old_flat))
        v_cat.index_copy_(1, new_flat, v_cat.index_select(1, old_flat))

        k_cat = k_cat.view(L, max_pages, page_len, num_heads, head_dim)
        v_cat = v_cat.view(L, max_pages, page_len, num_heads, head_dim)

        self.cache_data[:, :, 0, :, :, :].copy_(k_cat, non_blocking=True)
        self.cache_data[:, :, 1, :, :, :].copy_(v_cat, non_blocking=True)


class RequestKvCache:
    """Per-request KV cache manager. Tracks allocated pages and sequence length."""

    def __init__(self, kvCachePool: KvCachePool, page_len: int, seq_init_len: int):
        self.kvCachePool = kvCachePool
        self.page_len = page_len
        init_num_pages = math.ceil(seq_init_len / self.page_len) if seq_init_len > 0 else 0
        self.kv_last_page_len = seq_init_len - (init_num_pages - 1) * self.page_len if init_num_pages > 0 else 0
        self.kv_page_indices = kvCachePool.allocate(init_num_pages) if init_num_pages > 0 else []
        self.kv_len = seq_init_len
        self.is_released = False

    def get_seq_length(self):
        return self.kv_len

    def increment(self, num_tokens: int = 1):
        """Grow the sequence by num_tokens, allocating pages as needed."""
        if num_tokens <= 0:
            return
        # If no pages yet, allocate the first one
        if len(self.kv_page_indices) == 0:
            new_indices = self.kvCachePool.allocate(1)
            self.kv_page_indices.extend(new_indices)
            self.kv_last_page_len = 0
        self.kv_len += num_tokens
        self.kv_last_page_len += num_tokens
        while self.kv_last_page_len > self.page_len:
            self.kv_last_page_len -= self.page_len
            new_indices = self.kvCachePool.allocate(1)
            self.kv_page_indices.extend(new_indices)

    def decrement(self, num_tokens: int = 1):
        """Remove num_tokens from the end of the cache."""
        if num_tokens <= 0:
            return
        if num_tokens > self.kv_len:
            num_tokens = self.kv_len
        self.kv_len -= num_tokens
        needed_pages = math.ceil(self.kv_len / self.page_len) if self.kv_len > 0 else 0
        while len(self.kv_page_indices) > needed_pages:
            last_page = self.kv_page_indices.pop()
            self.kvCachePool.deallocate([last_page])
        if self.kv_len == 0:
            self.kv_last_page_len = 0
        else:
            self.kv_last_page_len = (self.kv_len - 1) % self.page_len + 1

    def release(self):
        """Return all pages to the pool."""
        if not self.is_released:
            self.kvCachePool.deallocate(self.kv_page_indices)
            self.is_released = True

    def reorder_cache_with_offset(
        self, beam_idx: torch.LongTensor, offset: int = 0, num_new_tokens: int = 0
    ):
        """Reorder cache for speculative decoding verification."""
        if offset != 0:
            offset -= 1
        self.kvCachePool.reorder_cache_with_offset(beam_idx, offset, num_new_tokens)
        self.kv_len = offset + beam_idx.size(0)
        if self.kv_len == 0:
            self.kv_last_page_len = 0
        else:
            self.kv_last_page_len = (self.kv_len - 1) % self.page_len + 1
        num_pages_needed = math.ceil(self.kv_len / self.page_len) if self.kv_len > 0 else 0
        current_num_pages = len(self.kv_page_indices)
        if current_num_pages > num_pages_needed:
            extra_pages = self.kv_page_indices[num_pages_needed:]
            self.kvCachePool.deallocate(extra_pages)
            self.kv_page_indices = self.kv_page_indices[:num_pages_needed]


def getKvCacheBatchPosition(
    request_kv_caches: List[RequestKvCache],
    mode: str,
    device: torch.device,
    treeTokens: int = 0,
) -> KvCacheBatchPosition:
    """
    Build a KvCacheBatchPosition from a list of RequestKvCache objects.

    Args:
        mode: "prefill", "decode", or "tree"
        treeTokens: number of tree tokens (only used in "tree" mode)
    """
    kv_page_indices_list = []
    kv_page_indptr_list = []
    seq_indptr_list = []
    kv_last_page_len_list = []
    seq_lens_list = []
    cum_pages = 0
    cum_seq_len = 0

    for request_kv_cache in request_kv_caches:
        kv_page_indices_list.extend(request_kv_cache.kv_page_indices)
        kv_page_indptr_list.append(cum_pages)
        seq_indptr_list.append(cum_seq_len)
        kv_last_page_len_list.append(request_kv_cache.kv_last_page_len)
        seq_lens_list.append(request_kv_cache.kv_len)
        cum_pages += len(request_kv_cache.kv_page_indices)

        if mode == "prefill":
            cum_seq_len += request_kv_cache.kv_len
        elif mode == "decode":
            cum_seq_len += 1
        elif mode == "tree":
            cum_seq_len += treeTokens
        else:
            raise ValueError(f"Invalid mode: {mode}")

    kv_page_indptr_list.append(cum_pages)
    seq_indptr_list.append(cum_seq_len)

    kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device=device)
    seq_indptr = torch.tensor(seq_indptr_list, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    kv_append_length = torch.tensor([cum_seq_len], dtype=torch.int32, device=device)
    kv_append_indptr = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.cumsum(kv_append_length, dim=0),
    ])

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr, seq_lens, cum_seq_len
    )

    return KvCacheBatchPosition(
        seq_indptr=seq_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        batch_indices=batch_indices,
        positions=positions,
    )


def copy_kv_pages(
    src_pool: KvCachePool,
    src_request: RequestKvCache,
    dst_pool: KvCachePool,
) -> RequestKvCache:
    """Copy KV cache pages from source to destination pool.

    Creates a new RequestKvCache in dst_pool with the same content as src_request.
    Handles dtype conversion (e.g. bf16 -> fp16) and different page sizes
    between source and destination pools automatically.

    Args:
        src_pool: Source KV cache pool (e.g. target model's pool)
        src_request: Source request cache to copy from
        dst_pool: Destination KV cache pool (e.g. draft model's pool)

    Returns:
        New RequestKvCache in dst_pool with copied KV data
    """
    seq_len = src_request.kv_len
    if seq_len == 0:
        return RequestKvCache(
            kvCachePool=dst_pool,
            page_len=dst_pool.page_len,
            seq_init_len=0,
        )

    src_page_len = src_pool.page_len
    dst_page_len = dst_pool.page_len

    dst_num_pages = math.ceil(seq_len / dst_page_len)
    dst_pages = dst_pool.allocate(dst_num_pages)

    if src_page_len == dst_page_len:
        # Fast path: 1:1 page copy
        for src_page, dst_page in zip(src_request.kv_page_indices, dst_pages):
            dst_pool.cache_data[:, dst_page].copy_(
                src_pool.cache_data[:, src_page].to(dst_pool.dtype),
                non_blocking=True,
            )
    else:
        # Different page sizes: copy token-by-token across page boundaries
        # cache_data shape: [num_layers, max_pages, 2, page_len, num_heads, head_dim]
        for tok in range(seq_len):
            src_page_idx = tok // src_page_len
            src_slot = tok % src_page_len
            dst_page_idx = tok // dst_page_len
            dst_slot = tok % dst_page_len
            src_phys = src_request.kv_page_indices[src_page_idx]
            dst_phys = dst_pages[dst_page_idx]
            dst_pool.cache_data[:, dst_phys, :, dst_slot].copy_(
                src_pool.cache_data[:, src_phys, :, src_slot].to(dst_pool.dtype),
                non_blocking=True,
            )

    # Build a RequestKvCache that mirrors the source's state
    dst_request = RequestKvCache(
        kvCachePool=dst_pool,
        page_len=dst_pool.page_len,
        seq_init_len=0,
    )
    dst_request.kv_page_indices = dst_pages
    dst_request.kv_len = seq_len
    dst_request.kv_last_page_len = (seq_len - 1) % dst_page_len + 1

    return dst_request
