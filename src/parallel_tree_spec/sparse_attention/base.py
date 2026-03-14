"""
Abstract base class and config for sparse attention strategies.

Strategies filter KV cache page indices during beam search decode steps
to reduce the attention budget. Prefill and tree verification always
use full attention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class SparseAttentionConfig:
    """Configuration for sparse attention during beam search decode."""
    enabled: bool = False
    method: str = "none"             # "none", "pillar"
    budget_ratio: float = 0.05       # fraction of KV pages to keep
    min_budget_tokens: int = 128     # minimum tokens to keep
    recent_window: int = 32          # always include recent N tokens
    importance_method: str = "kv_norm"  # "kv_norm" or "qk_score"


class SparseAttentionStrategy(ABC):
    """Base class for sparse attention page filtering strategies."""

    def __init__(self, config: SparseAttentionConfig):
        self.config = config

    @abstractmethod
    def filter_pages(
        self,
        all_pages: List[int],
        seq_len: int,
        page_len: int,
    ) -> List[int]:
        """Return subset of pages to attend to during decode.

        Args:
            all_pages: Full ordered list of page indices for this beam.
            seq_len: Current sequence length (number of tokens written).
            page_len: Tokens per page.

        Returns:
            Filtered list of page indices (must preserve original order).
        """

    @abstractmethod
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
        """Compute and store importance scores (called after prefill).

        Args:
            kv_cache: Full KV cache tensor [num_layers, max_pages, 2, page_len, num_heads, head_dim].
            page_indices: Page indices used by the prompt.
            seq_len: Prompt length in tokens.
            page_len: Tokens per page.
            num_kv_heads: Number of KV attention heads.
            head_dim: Dimension per head.
            q: Optional query tensor for qk_score method [num_heads, head_dim]
               (last token's query vectors, after RoPE).
        """

    @abstractmethod
    def reset(self):
        """Clear state for new prompt."""
