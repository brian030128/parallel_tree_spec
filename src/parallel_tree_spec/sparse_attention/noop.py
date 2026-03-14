"""
No-op sparse attention strategy — full attention passthrough (baseline).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .base import SparseAttentionConfig, SparseAttentionStrategy


class NoopStrategy(SparseAttentionStrategy):
    """Passthrough strategy that returns all pages unchanged."""

    def __init__(self, config: SparseAttentionConfig):
        super().__init__(config)

    def filter_pages(
        self,
        all_pages: List[int],
        seq_len: int,
        page_len: int,
    ) -> List[int]:
        return all_pages

    def update_importance(
        self,
        kv_cache: torch.Tensor,
        page_indices: List[int],
        seq_len: int,
        page_len: int,
        num_kv_heads: int,
        head_dim: int,
        q: Optional[torch.Tensor] = None,
    ):
        pass

    def reset(self):
        pass
