"""
Tree verification for speculative decoding.

Implements exact verification (greedy argmax matching) of draft trees
against target model logits, plus the target model tree decoding forward pass.

Adapted from:
  - subspec_v2/specdecodes/models/utils/tree_verify.py (exact method)
  - subspec_v2/specdecodes/models/generators/classic_sd_fi.py (_tree_decoding)
  - subspec_v2/specdecodes/models/generators/classic_sd.py (_verify_step, _sample_token)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .tree import Tree
from .flashinfer.cache_manager import (
    KvCachePool,
    RequestKvCache,
    getKvCacheBatchPosition,
)
from .flashinfer.attention_wrapper import BeFlashinferWrapper


@dataclass
class VerifyResult:
    """Result of one draft-verify cycle."""
    sampled_tokens: torch.Tensor   # [1, L] accepted + bonus token
    hidden_indices: torch.Tensor   # [L] indices into tree (for KV reorder)
    total_len: int                 # total verification steps attempted
    accept_len: int                # accepted tokens (excludes bonus)
    verify_time: float             # seconds for target forward + verify


# ---------------------------------------------------------------------------
# Callbacks (equivalent to GeneratorBase._sample_token and _verify_step)
# ---------------------------------------------------------------------------

def _sample_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities (greedy path: just softmax).

    Args:
        logits: [batch, seq_len, vocab_size] or [seq_len, vocab_size]

    Returns:
        probs: same shape, softmax over last dim
    """
    return torch.softmax(logits, dim=-1)


def _verify_step_greedy(
    probs: torch.Tensor,
    children_token_ids: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Greedy verify step: argmax of target probs, check if it matches any child.

    Args:
        probs: [vocab_size] probability distribution at this node
        children_token_ids: [C] token IDs of tree children

    Returns:
        (accepted_token, bonus_token):
            - If accepted: (token_id, None)
            - If rejected: (None, sampled_token_id)
    """
    sampled_token_id = probs.argmax()
    if torch.any(sampled_token_id == children_token_ids):
        return sampled_token_id, None
    else:
        return None, sampled_token_id


# ---------------------------------------------------------------------------
# Tree input preparation
# ---------------------------------------------------------------------------

def prepare_tree_inputs(
    tree: Tree,
    position_offset: int,
    device: torch.device,
    skip_nodes: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare tree inputs for target model forward.

    Args:
        tree: Draft tree from beam search
        position_offset: Position of root token in the full sequence
        device: Target device
        skip_nodes: Number of leading nodes to skip

    Returns:
        tree_input_ids: [N] token IDs for tree nodes
        tree_position_ids: [N] position IDs (depth + offset)
        tree_mask: [1, 1, N, prefix_length + N] boolean attention mask
    """
    node_data = tree.get_tree_data(skip_nodes)
    tree_input_ids = node_data["token_ids"].to(device)
    tree_position_ids = (node_data["depths"] + position_offset).to(device)

    # FlashInfer uses non-inverted boolean mask (True = can attend)
    tree_mask = tree.create_attention_mask(
        prefix_length=position_offset, skip_nodes=skip_nodes, device=device
    )

    return tree_input_ids, tree_position_ids, tree_mask


# ---------------------------------------------------------------------------
# Target model tree decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def target_tree_decode(
    target_model: torch.nn.Module,
    flashinfer_wrapper: BeFlashinferWrapper,
    tree: Tree,
    request_kv_cache: RequestKvCache,
    position_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Run target model forward on the draft tree using FlashInfer tree attention.

    Args:
        target_model: Full-precision target model (FlashInfer-patched)
        flashinfer_wrapper: FlashInfer wrapper for target model
        tree: Draft tree from beam search
        request_kv_cache: Target model's KV cache
        position_offset: Position of tree root in full sequence
        device: Device

    Returns:
        logits: [1, N, vocab_size] target model logits for tree nodes
    """
    kv_cache_pool = request_kv_cache.kvCachePool
    num_tokens = tree.current_size

    tree_input_ids, tree_position_ids, tree_mask = prepare_tree_inputs(
        tree, position_offset=position_offset, device=device
    )

    # Allocate KV space for tree tokens
    request_kv_cache.increment(num_tokens)

    batch_position = getKvCacheBatchPosition(
        request_kv_caches=[request_kv_cache],
        mode="tree",
        device=device,
        treeTokens=num_tokens,
    )

    flashinfer_wrapper.prepareAttention(
        "tree",
        batch_position,
        kv_cache_pool.page_len,
        "NONE",
        kv_cache_pool.cache_data[0].dtype,
        attention_mask=tree_mask,
    )

    outputs = target_model(
        input_ids=tree_input_ids.unsqueeze(0),
        past_key_values=None,
        position_ids=tree_position_ids.unsqueeze(0),
        use_cache=False,
        kvCachePool=kv_cache_pool,
        batch_position=batch_position,
        mode="tree",
        flashinferWrapper=flashinfer_wrapper,
    )

    return outputs.logits


# ---------------------------------------------------------------------------
# Exact tree verification
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_tree_exact(
    tree: Tree,
    logits: torch.Tensor,
    root_ind: int = 0,
    eos_token_id: Optional[int] = None,
    skip_nodes: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Exact verification: greedy argmax matching along tree paths.

    Walks from root, at each node checks if target's argmax matches any child.
    Accepts the matching child and continues. Appends a bonus token at the end.

    Args:
        tree: Draft tree
        logits: [1, N, vocab_size] target model logits for tree nodes
        root_ind: Root node index
        eos_token_id: EOS token ID (stop if accepted)
        skip_nodes: Number of leading nodes skipped

    Returns:
        sampled_tokens: [1, L] accepted tokens + bonus
        hidden_indices: [L] tree node indices (for KV cache reorder)
        total_len: total verification steps
        accept_len: accepted tokens (excludes bonus)
    """
    # Get probabilities from logits
    global_p = _sample_probs(logits)
    global_p = global_p.squeeze(0).cpu()  # [N, vocab_size]

    sampled_tokens = torch.empty(0, dtype=torch.long, device="cpu")
    hidden_indices = torch.empty(0, dtype=torch.long, device="cpu")
    total_len = 0
    accept_len = 0

    node_data = tree.get_tree_data(skip_nodes=skip_nodes)
    token_ids = node_data["token_ids"]

    cur_ind = torch.tensor([int(root_ind)], dtype=torch.long, device="cpu")
    children_inds = tree.get_children_indices(cur_ind)
    children_token_ids = token_ids[children_inds - int(skip_nodes)]

    bonus_token = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    while children_inds.numel() > 0:
        total_len += 1
        dist = global_p[cur_ind - int(skip_nodes)].squeeze(0)
        accept_token_id, bonus_token = _verify_step_greedy(dist, children_token_ids)

        if accept_token_id is not None:
            accept_len += 1
            sampled_tokens = torch.cat([sampled_tokens, accept_token_id[None]])
            hidden_indices = torch.cat([hidden_indices, cur_ind])

            if eos_token_id is not None and int(accept_token_id) == int(eos_token_id):
                break

            cur_ind = children_inds[children_token_ids == accept_token_id]
            children_inds = tree.get_children_indices(cur_ind)
            children_token_ids = token_ids[children_inds - int(skip_nodes)]

            if (children_inds - int(skip_nodes)).numel() == 0:
                break
            if int(torch.min(children_inds - int(skip_nodes)).item()) >= int(global_p.shape[0]):
                break
        else:
            break

    # Bonus token (unless EOS already emitted)
    if sampled_tokens.numel() == 0 or (
        eos_token_id is None or int(sampled_tokens[-1].item()) != int(eos_token_id)
    ):
        if bonus_token is None:
            dist = global_p[cur_ind - int(skip_nodes)].squeeze(0)
            bonus_token = dist.argmax()

        if bonus_token is not None:
            sampled_tokens = torch.cat([sampled_tokens, bonus_token.unsqueeze(0)])
            hidden_indices = torch.cat([hidden_indices, cur_ind])

    return sampled_tokens.unsqueeze(0), hidden_indices, total_len, accept_len


# ---------------------------------------------------------------------------
# Combined: target decode + verify
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_draft_tree(
    target_model: torch.nn.Module,
    flashinfer_wrapper: BeFlashinferWrapper,
    tree: Tree,
    request_kv_cache: RequestKvCache,
    position_offset: int,
    device: torch.device,
    eos_token_id: Optional[int] = None,
) -> VerifyResult:
    """Run full verification: target model tree decode + exact verify.

    Args:
        target_model: Full-precision target model
        flashinfer_wrapper: FlashInfer wrapper for target model
        tree: Draft tree from beam search
        request_kv_cache: Target model's KV cache
        position_offset: Position of tree root in full sequence
        device: Device
        eos_token_id: EOS token ID

    Returns:
        VerifyResult with accepted tokens, indices, counts, and timing
    """
    t_start = time.perf_counter()

    # Target model forward on tree
    logits = target_tree_decode(
        target_model, flashinfer_wrapper, tree,
        request_kv_cache, position_offset, device,
    )

    # Exact verification
    sampled_tokens, hidden_indices, total_len, accept_len = verify_tree_exact(
        tree, logits, root_ind=0, eos_token_id=eos_token_id,
    )

    t_end = time.perf_counter()

    return VerifyResult(
        sampled_tokens=sampled_tokens,
        hidden_indices=hidden_indices,
        total_len=total_len,
        accept_len=accept_len,
        verify_time=t_end - t_start,
    )
