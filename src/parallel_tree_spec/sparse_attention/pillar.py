"""
Pillar sparse attention strategy — Top-K page selection by importance.

Selects the most important KV pages based on either:
- kv_norm: Sum of ||K||^2 across layers and heads per token, aggregated to pages.
- qk_score: Q @ K^T dot product scores using the last prefill query vectors.

Always includes recent pages (sliding window) to preserve local context.
"""

from __future__ import annotations

import math
from typing import List, Optional, Set

import torch

from .base import SparseAttentionConfig, SparseAttentionStrategy


class PillarStrategy(SparseAttentionStrategy):
    """Top-K page selection based on importance scores."""

    def __init__(self, config: SparseAttentionConfig):
        super().__init__(config)
        self._important_pages: Optional[Set[int]] = None

    def reset(self):
        self._important_pages = None

    def filter_pages(
        self,
        all_pages: List[int],
        seq_len: int,
        page_len: int,
    ) -> List[int]:
        # Cold start: no importance computed yet → full attention
        if self._important_pages is None:
            return all_pages

        num_pages = len(all_pages)
        if num_pages <= 1:
            return all_pages

        # Compute budget in pages
        budget_tokens = max(
            int(seq_len * self.config.budget_ratio),
            self.config.min_budget_tokens,
        )
        budget_pages = max(1, math.ceil(budget_tokens / page_len))

        # Recent window pages (always included)
        recent_tokens = self.config.recent_window
        recent_pages_count = max(1, math.ceil(recent_tokens / page_len))
        # Recent pages are the last N pages in the list
        recent_page_set = set(all_pages[-recent_pages_count:])

        # Important pages (from stored set)
        important_set = self._important_pages | recent_page_set

        # If budget already covers everything, return all
        if budget_pages >= num_pages:
            return all_pages

        # Select: union of important + recent, capped at budget
        # Important pages are already ranked; take up to budget
        selected = set()
        # Always add recent
        selected.update(recent_page_set)
        # Fill remaining budget from important pages
        remaining_budget = budget_pages - len(selected)
        if remaining_budget > 0:
            for p in all_pages:
                if p in self._important_pages and p not in selected:
                    selected.add(p)
                    remaining_budget -= 1
                    if remaining_budget <= 0:
                        break

        # Preserve original order
        return [p for p in all_pages if p in selected]

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
        """Compute page importance and store the top-K page set.

        Args:
            kv_cache: [num_layers, max_pages, 2, page_len, num_heads, head_dim]
            page_indices: Prompt's page indices.
            seq_len: Prompt length in tokens.
            page_len: Tokens per page.
            num_kv_heads: Number of KV heads.
            head_dim: Head dimension.
            q: Optional [num_q_heads, head_dim] query for qk_score method.
        """
        if not page_indices:
            self._important_pages = None
            return

        num_pages = len(page_indices)
        page_idx_tensor = torch.tensor(page_indices, dtype=torch.long, device=kv_cache.device)

        if self.config.importance_method == "qk_score" and q is not None:
            self._compute_qk_importance(kv_cache, page_idx_tensor, seq_len, page_len, q)
        else:
            self._compute_kv_norm_importance(kv_cache, page_idx_tensor, seq_len, page_len)

    def _compute_kv_norm_importance(
        self,
        kv_cache: torch.Tensor,
        page_idx_tensor: torch.Tensor,
        seq_len: int,
        page_len: int,
    ):
        """KV-norm proxy: sum ||K||^2 across layers and heads per token, then aggregate to pages."""
        # kv_cache: [num_layers, max_pages, 2, page_len, num_heads, head_dim]
        # Extract K for selected pages: [layers, pages, page_len, heads, dim]
        keys = kv_cache[:, page_idx_tensor, 0]  # [layers, num_pages, page_len, heads, dim]

        # ||K||^2 per token position, summed across layers and heads
        # [layers, num_pages, page_len, heads] -> [num_pages, page_len]
        k_norms_sq = keys.float().pow(2).sum(dim=-1).sum(dim=(0, 3))  # [num_pages, page_len]

        # Mask out unfilled positions in the last page
        num_pages = page_idx_tensor.shape[0]
        last_page_fill = seq_len % page_len
        if last_page_fill == 0 and seq_len > 0:
            last_page_fill = page_len
        if last_page_fill < page_len and num_pages > 0:
            k_norms_sq[-1, last_page_fill:] = 0.0

        # Page-level importance: sum across token positions within each page
        page_scores = k_norms_sq.sum(dim=1)  # [num_pages]

        self._store_topk_pages(page_scores, page_idx_tensor, seq_len, page_len)

    def _compute_qk_importance(
        self,
        kv_cache: torch.Tensor,
        page_idx_tensor: torch.Tensor,
        seq_len: int,
        page_len: int,
        q: torch.Tensor,
    ):
        """Q*K scoring: compute Q @ K^T for each KV position, aggregate to pages.

        Args:
            q: [num_q_heads, head_dim] — last token's query vectors.
        """
        # keys: [layers, num_pages, page_len, kv_heads, dim]
        keys = kv_cache[:, page_idx_tensor, 0]
        num_layers, num_pages, pl, num_kv_heads, head_dim = keys.shape

        # q is [num_q_heads, head_dim]. If GQA, num_q_heads > num_kv_heads.
        # Average Q across the GQA group to match KV heads.
        num_q_heads = q.shape[0]
        if num_q_heads != num_kv_heads:
            group_size = num_q_heads // num_kv_heads
            q_grouped = q.view(num_kv_heads, group_size, head_dim).mean(dim=1)  # [kv_heads, dim]
        else:
            q_grouped = q  # [kv_heads, dim]

        # Compute attention scores: for each layer, Q @ K^T
        # keys: [layers, pages, page_len, kv_heads, dim]
        # q_grouped: [kv_heads, dim]
        # Score per position: sum across layers and heads of (q · k)
        q_expanded = q_grouped.float()  # [kv_heads, dim]
        keys_flat = keys.float()  # [layers, pages, page_len, kv_heads, dim]

        # Dot product: [layers, pages, page_len, kv_heads]
        scores = (keys_flat * q_expanded.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        # Aggregate across layers and heads: [pages, page_len]
        scores = scores.sum(dim=(0, 3))

        # Mask unfilled positions in last page
        num_pages_val = page_idx_tensor.shape[0]
        last_page_fill = seq_len % page_len
        if last_page_fill == 0 and seq_len > 0:
            last_page_fill = page_len
        if last_page_fill < page_len and num_pages_val > 0:
            scores[-1, last_page_fill:] = 0.0

        # Page-level importance
        page_scores = scores.sum(dim=1)  # [pages]

        self._store_topk_pages(page_scores, page_idx_tensor, seq_len, page_len)

    def _store_topk_pages(
        self, page_scores: torch.Tensor, page_idx_tensor: torch.Tensor,
        seq_len: int, page_len: int,
    ):
        """Select top-K pages by score and store as a set."""
        num_pages = page_scores.shape[0]

        # Budget: same logic as filter_pages for consistency
        budget_tokens = max(
            int(seq_len * self.config.budget_ratio),
            self.config.min_budget_tokens,
        )
        budget_pages = max(1, math.ceil(budget_tokens / page_len))
        budget_pages = min(budget_pages, num_pages)

        # Top-K by score
        topk_count = min(budget_pages, num_pages)
        _, topk_indices = page_scores.topk(topk_count)
        # Map back to physical page indices
        selected_physical = page_idx_tensor[topk_indices].tolist()
        self._important_pages = set(selected_physical)
