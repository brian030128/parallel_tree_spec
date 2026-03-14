"""
Tree verification for speculative decoding.

Implements exact verification (greedy argmax matching) and traversal
verification (Weng et al., arXiv:2505.12398) of draft trees against
target model logits, plus the target model tree decoding forward pass.

Adapted from:
  - subspec_v2/specdecodes/models/utils/tree_verify.py (exact method)
  - subspec_v2/specdecodes/models/utils/traversal_verification.py (traversal method)
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
        prefix_length=position_offset + 1, skip_nodes=skip_nodes, device=device
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
# Traversal verification (Weng et al., arXiv:2505.12398)
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_tree_traversal(
    tree: Tree,
    logits: torch.Tensor,
    root_ind: int = 0,
    eos_token_id: Optional[int] = None,
    skip_nodes: int = 0,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Traversal verification: stochastic acceptance with residual resampling.

    Uses sequence-level acceptance with bottom-up leaf-first traversal and
    residual distribution updates.  Achieves higher acceptance rates than
    exact (greedy argmax) verification.

    Args:
        tree: Draft tree
        logits: [1, N, vocab_size] target model logits for tree nodes
        root_ind: Root node index
        eos_token_id: EOS token ID (stop if accepted)
        skip_nodes: Number of leading nodes skipped
        do_sample: If True, sample bonus token via multinomial; else argmax

    Returns:
        sampled_tokens: [1, L] accepted tokens + bonus
        hidden_indices: [L] tree node indices (for KV cache reorder)
        total_len: total tokens in output (accept_len + bonus)
        accept_len: accepted tokens (excludes bonus)
    """
    eps = 1e-8
    dtype = torch.float32

    # ── 1. Flatten tree to tensors ──────────────────────────────────────────
    # Apply same temperature as draft model so M_b and M_s are comparable
    scaled_logits = logits / temperature if temperature != 1.0 else logits
    global_p = _sample_probs(scaled_logits).squeeze(0).cpu()  # [N, vocab_size]

    relevant_nodes = tree.nodes[skip_nodes:]
    num_nodes = len(relevant_nodes)
    tree_idx_map = {i + skip_nodes: i for i in range(num_nodes)}

    t_token_ids = torch.empty(num_nodes, dtype=torch.long)
    t_parent_indices = torch.full((num_nodes,), -1, dtype=torch.long)
    t_depths = torch.zeros(num_nodes, dtype=torch.long)
    t_cum_probs = torch.zeros(num_nodes, dtype=dtype)
    local_adj: List[List[int]] = [[] for _ in range(num_nodes)]
    stack = [(root_ind, 0)]

    while stack:
        orig_idx, depth = stack.pop()
        if orig_idx < skip_nodes:
            continue
        local_idx = tree_idx_map[orig_idx]
        node = tree.nodes[orig_idx]

        t_token_ids[local_idx] = node.token_id
        t_cum_probs[local_idx] = node.cumulative_probability
        t_depths[local_idx] = depth
        if node.parent is not None and node.parent >= skip_nodes:
            parent_local = tree_idx_map[node.parent]
            t_parent_indices[local_idx] = parent_local
            local_adj[parent_local].append(local_idx)

        for child_idx in reversed(node.children):
            stack.append((child_idx, depth + 1))

    # ── 2. Compute M_b, M_s, p_alpha ───────────────────────────────────────
    t_Mb_scalars = torch.zeros(num_nodes, dtype=dtype)
    t_Mb_scalars[0] = 1.0

    non_roots = t_parent_indices != -1
    parents_of_non_roots = t_parent_indices[non_roots]
    tokens_of_non_roots = t_token_ids[non_roots]
    t_Mb_scalars[non_roots] = global_p[parents_of_non_roots, tokens_of_non_roots].to(dtype)

    t_Ms_scalars = torch.zeros(num_nodes, dtype=dtype)
    t_Ms_scalars[0] = 1.0
    t_Ms_scalars[non_roots] = t_cum_probs[non_roots]

    t_p_alpha = torch.zeros(num_nodes, dtype=dtype)
    t_p_alpha[0] = 1.0

    max_depth = int(t_depths.max().item())
    for d in range(1, max_depth + 1):
        mask = t_depths == d
        if not mask.any():
            continue
        node_indices = torch.nonzero(mask).squeeze(-1)
        parents = t_parent_indices[node_indices]
        p_parents = t_p_alpha[parents]
        mb = t_Mb_scalars[node_indices]
        ms = t_Ms_scalars[node_indices]
        ratio = torch.zeros_like(mb)
        valid_ms = ms > eps
        ratio[valid_ms] = mb[valid_ms] / ms[valid_ms]
        t_p_alpha[node_indices] = torch.minimum(p_parents * ratio, torch.tensor(1.0))

    # ── 3. Traversal loop ──────────────────────────────────────────────────
    valid_mask = torch.ones(num_nodes, dtype=torch.bool)
    accepted_node_idx = 0

    while True:
        # Walk from root to first leaf (first valid child at each level)
        curr = 0
        while True:
            first_child = None
            for c in local_adj[curr]:
                if valid_mask[c]:
                    first_child = c
                    break
            if first_child is None:
                break
            curr = first_child
        candidate_idx = curr

        if not valid_mask[candidate_idx]:
            break

        # Stochastic accept/reject
        p_val = t_p_alpha[candidate_idx].item()
        eta = torch.rand(1).item()

        if eta < p_val:
            accepted_node_idx = candidate_idx
            break

        # Reject: mark deleted, update siblings
        valid_mask[candidate_idx] = False
        old_ms_rejected = t_Ms_scalars[candidate_idx].item()
        t_Ms_scalars[candidate_idx] = 0.0
        parent_idx = t_parent_indices[candidate_idx].item()

        if parent_idx == -1:
            accepted_node_idx = 0
            break

        siblings = torch.tensor(local_adj[parent_idx], dtype=torch.long)
        if len(siblings) == 0:
            continue

        active_sibling_mask = valid_mask[siblings]
        if not active_sibling_mask.any():
            continue
        active_siblings = siblings[active_sibling_mask]

        ms_rejected = old_ms_rejected
        p_old = t_p_alpha[parent_idx].item()

        ms_denom = 1.0 - ms_rejected
        if ms_denom < eps:
            ms_denom = eps

        # Compute S (total residual mass)
        all_child_indices = torch.tensor(local_adj[parent_idx], dtype=torch.long)
        mbs = t_Mb_scalars[all_child_indices]
        mss = t_Ms_scalars[all_child_indices]
        S_tree = torch.clamp(p_old * mbs - mss, min=0).sum().item()
        S_outside = p_old * (1.0 - mbs.sum().item())
        S = S_tree + S_outside
        if S < eps:
            S = eps

        # Update p_alpha for parent
        p_new = S / (S + 1.0 - p_old)
        t_p_alpha[parent_idx] = p_new

        # Update Ms and Mb for active siblings
        sib_mbs = t_Mb_scalars[active_siblings]
        sib_mss = t_Ms_scalars[active_siblings]
        new_ms_sibs = sib_mss / ms_denom
        new_mb_sibs = torch.clamp(p_old * sib_mbs - sib_mss, min=0) / S
        t_Ms_scalars[active_siblings] = new_ms_sibs
        t_Mb_scalars[active_siblings] = new_mb_sibs

        # Update rejected node's Mb to its residual value
        old_mb_rejected = t_Mb_scalars[candidate_idx].item()
        t_Mb_scalars[candidate_idx] = max(p_old * old_mb_rejected - old_ms_rejected, 0.0) / S

        # Update p_alpha for active siblings
        ratio_sibs = torch.zeros_like(new_mb_sibs)
        valid_ms_sibs = new_ms_sibs > eps
        ratio_sibs[valid_ms_sibs] = new_mb_sibs[valid_ms_sibs] / new_ms_sibs[valid_ms_sibs]
        t_p_alpha[active_siblings] = torch.minimum(p_new * ratio_sibs, torch.tensor(1.0))

        # Propagate p_alpha updates to descendants
        update_queue = active_siblings.tolist()
        while update_queue:
            u = update_queue.pop(0)
            p_u = t_p_alpha[u].item()
            u_children = local_adj[u]
            if not u_children:
                continue
            u_children_t = torch.tensor(u_children, dtype=torch.long)
            valid_children = u_children_t[valid_mask[u_children_t]]
            if len(valid_children) > 0:
                c_mb = t_Mb_scalars[valid_children]
                c_ms = t_Ms_scalars[valid_children]
                c_ratio = torch.zeros_like(c_mb)
                c_valid_ms = c_ms > eps
                c_ratio[c_valid_ms] = c_mb[c_valid_ms] / c_ms[c_valid_ms]
                t_p_alpha[valid_children] = torch.minimum(p_u * c_ratio, torch.tensor(1.0))
                update_queue.extend(valid_children.tolist())

    # ── 4. Reconstruct output ──────────────────────────────────────────────
    path_indices = []
    curr = accepted_node_idx
    while curr != -1:
        path_indices.append(curr)
        curr = t_parent_indices[curr].item()
    path_indices.reverse()

    sampled_tokens_list = []
    hidden_indices_list = []

    for i in range(1, len(path_indices)):
        local_idx = path_indices[i]
        token = t_token_ids[local_idx].item()
        orig_idx = local_idx + skip_nodes
        sampled_tokens_list.append(token)
        hidden_indices_list.append(orig_idx)
        if eos_token_id is not None and token == eos_token_id:
            break

    # Bonus token
    should_sample_bonus = True
    if sampled_tokens_list and eos_token_id is not None and sampled_tokens_list[-1] == eos_token_id:
        should_sample_bonus = False

    if should_sample_bonus:
        leaf_local_idx = accepted_node_idx
        if leaf_local_idx < global_p.shape[0]:
            dist = global_p[leaf_local_idx]
            if do_sample:
                bonus_token = torch.multinomial(dist, 1).item()
            else:
                bonus_token = torch.argmax(dist).item()
        else:
            bonus_token = 0
        sampled_tokens_list.append(bonus_token)
        hidden_indices_list.append(leaf_local_idx + skip_nodes)

    device = logits.device
    ret_tokens = torch.tensor([sampled_tokens_list], dtype=torch.long, device=device)
    ret_indices = torch.tensor(hidden_indices_list, dtype=torch.long, device=device)

    total_len = len(sampled_tokens_list)
    accept_len = max(0, total_len - 1)

    return ret_tokens, ret_indices, total_len, accept_len


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
    verification_method: str = "traversal",
    temperature: float = 1.0,
) -> VerifyResult:
    """Run full verification: target model tree decode + verify.

    Args:
        target_model: Full-precision target model
        flashinfer_wrapper: FlashInfer wrapper for target model
        tree: Draft tree from beam search
        request_kv_cache: Target model's KV cache
        position_offset: Position of tree root in full sequence
        device: Device
        eos_token_id: EOS token ID
        verification_method: "traversal" (stochastic, higher acceptance) or
                             "exact" (greedy argmax matching)

    Returns:
        VerifyResult with accepted tokens, indices, counts, and timing
    """
    t_start = time.perf_counter()

    # Target model forward on tree
    logits = target_tree_decode(
        target_model, flashinfer_wrapper, tree,
        request_kv_cache, position_offset, device,
    )

    # Dispatch verification method
    if verification_method == "traversal":
        sampled_tokens, hidden_indices, total_len, accept_len = verify_tree_traversal(
            tree, logits, root_ind=0, eos_token_id=eos_token_id,
            temperature=temperature,
        )
    elif verification_method == "exact":
        sampled_tokens, hidden_indices, total_len, accept_len = verify_tree_exact(
            tree, logits, root_ind=0, eos_token_id=eos_token_id,
        )
    else:
        raise ValueError(f"Unknown verification method: {verification_method!r} (expected 'traversal' or 'exact')")

    t_end = time.perf_counter()

    return VerifyResult(
        sampled_tokens=sampled_tokens,
        hidden_indices=hidden_indices,
        total_len=total_len,
        accept_len=accept_len,
        verify_time=t_end - t_start,
    )
