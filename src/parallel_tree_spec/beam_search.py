"""
Beam search with copy-on-write paged KV cache using FlashInfer.

Runs K-beam search on a (possibly quantized) draft model, returning
a Tree of candidate tokens. Beams share prompt KV pages via COW.

Adapted from subspec_v2/specdecodes/models/draft_models/be_classic_sd_fi.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .tree import Tree, TreeNode
from .flashinfer.cache_manager import (
    KvCacheBatchPosition,
    KvCachePool,
    RequestKvCache,
    getKvCacheBatchPosition,
)
from .flashinfer.attention_wrapper import BeFlashinferWrapper


@dataclass
class BeamSearchConfig:
    """Configuration for beam search."""
    topk_len: int = 6          # beam width K
    max_depth: int = 10        # number of decode steps
    temperature: float = 1.0   # softmax temperature
    use_cuda_graph: bool = False  # capture decode steps in a CUDA graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_block(kvCachePool: KvCachePool, src_page: int, off: int) -> int:
    """COW: allocate a fresh page and copy the first `off` token slots from src_page."""
    new_page = kvCachePool.allocate(1)[0]
    kvCachePool.cache_data[:, new_page, :, :off] = kvCachePool.cache_data[:, src_page, :, :off]
    return new_page


def _build_beam_batch_position(
    beam_pages_list: List[List[int]],
    current_pos: int,
    page_size: int,
    device: torch.device,
) -> KvCacheBatchPosition:
    """Build KvCacheBatchPosition for K beams each writing one token at current_pos."""
    K = len(beam_pages_list)
    kv_page_indices = []
    kv_page_indptr = [0]
    for pages in beam_pages_list:
        kv_page_indices.extend(pages)
        kv_page_indptr.append(len(kv_page_indices))

    kv_last_page_len_val = current_pos % page_size + 1

    return KvCacheBatchPosition(
        seq_indptr=torch.arange(K + 1, dtype=torch.int32, device=device),
        kv_page_indptr=torch.tensor(kv_page_indptr, dtype=torch.int32, device=device),
        kv_page_indices=torch.tensor(kv_page_indices, dtype=torch.int32, device=device),
        kv_last_page_len=torch.tensor([kv_last_page_len_val] * K, dtype=torch.int32, device=device),
        batch_indices=torch.arange(K, dtype=torch.int32, device=device),
        positions=torch.tensor([current_pos] * K, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# CUDA graph runner
# ---------------------------------------------------------------------------

class CudaGraphRunner:
    """Captures and replays a CUDA graph for beam search decode steps."""

    def __init__(self, K: int, max_num_pages: int, device: torch.device):
        self.K = K
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_output = None

        # Static input buffers
        self.input_ids_buf = torch.zeros((K, 1), dtype=torch.long, device=device)
        self.position_ids_buf = torch.zeros((K, 1), dtype=torch.long, device=device)

        # Static batch position buffers (reused across replays)
        self.batch_position = KvCacheBatchPosition(
            seq_indptr=torch.arange(K + 1, dtype=torch.int32, device=device),
            kv_page_indptr=torch.zeros(K + 1, dtype=torch.int32, device=device),
            kv_page_indices=torch.zeros(K * max_num_pages, dtype=torch.int32, device=device),
            kv_last_page_len=torch.zeros(K, dtype=torch.int32, device=device),
            batch_indices=torch.arange(K, dtype=torch.int32, device=device),
            positions=torch.zeros(K, dtype=torch.int32, device=device),
        )

    def capture(
        self,
        model: torch.nn.Module,
        flashinfer_wrapper: "BeFlashinferWrapper",
        kvCachePool: KvCachePool,
        page_size: int,
        dtype: torch.dtype,
    ):
        """Warm up the model and capture a CUDA graph for decode forward."""
        K = self.K
        device = self.device

        # Allocate K temp pages for warmup
        temp_pages = kvCachePool.allocate(K)

        # Fill staging buffers pointing to temp pages (1 page per beam)
        self.input_ids_buf.fill_(0)
        self.position_ids_buf.fill_(0)
        indptr = torch.arange(K + 1, dtype=torch.int32, device=device)
        self.batch_position.kv_page_indptr.copy_(indptr)
        self.batch_position.kv_page_indices[:K].copy_(
            torch.tensor(temp_pages, dtype=torch.int32, device=device)
        )
        self.batch_position.kv_last_page_len.fill_(1)
        self.batch_position.positions.fill_(0)

        # 2 warmup forwards (populate internal buffers, JIT paths, etc.)
        for _ in range(2):
            flashinfer_wrapper.prepareAttention(
                "decode", self.batch_position, page_size, "NONE", dtype,
            )
            model(
                self.input_ids_buf,
                position_ids=self.position_ids_buf,
                past_key_values=None,
                use_cache=False,
                kvCachePool=kvCachePool,
                batch_position=self.batch_position,
                mode="decode",
                flashinferWrapper=flashinfer_wrapper,
            )

        # Capture
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        flashinfer_wrapper.prepareAttention(
            "decode", self.batch_position, page_size, "NONE", dtype,
        )
        with torch.cuda.graph(self.graph):
            self.static_output = model(
                self.input_ids_buf,
                position_ids=self.position_ids_buf,
                past_key_values=None,
                use_cache=False,
                kvCachePool=kvCachePool,
                batch_position=self.batch_position,
                mode="decode",
                flashinferWrapper=flashinfer_wrapper,
            )

        # Free temp pages
        kvCachePool.deallocate(temp_pages)

    def replay(
        self,
        beam_input_ids: torch.Tensor,
        beam_position_ids: torch.Tensor,
        batch_position: KvCacheBatchPosition,
        flashinfer_wrapper: "BeFlashinferWrapper",
        page_size: int,
        dtype: torch.dtype,
    ):
        """Copy real data into staging buffers, plan attention, replay graph."""
        # Copy model inputs
        self.input_ids_buf.copy_(beam_input_ids)
        self.position_ids_buf.copy_(beam_position_ids)

        # Copy batch position fields
        self.batch_position.kv_page_indptr.copy_(batch_position.kv_page_indptr)
        n_idx = batch_position.kv_page_indices.shape[0]
        self.batch_position.kv_page_indices[:n_idx].copy_(batch_position.kv_page_indices)
        self.batch_position.kv_last_page_len.copy_(batch_position.kv_last_page_len)
        self.batch_position.positions.copy_(batch_position.positions)

        # Plan attention (updates FlashInfer's pre-allocated decode buffers)
        flashinfer_wrapper.prepareAttention(
            "decode", self.batch_position, page_size, "NONE", dtype,
        )

        self.graph.replay()
        return self.static_output


# ---------------------------------------------------------------------------
# Main beam search
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(
    model: torch.nn.Module,
    request_kv_cache: RequestKvCache,
    input_ids: torch.Tensor,
    config: BeamSearchConfig,
    flashinfer_wrapper: BeFlashinferWrapper,
    prefilled_logits: Optional[torch.Tensor] = None,
    cuda_graph_runner: Optional["CudaGraphRunner"] = None,
) -> Tuple[Tree, List[float], Optional["CudaGraphRunner"]]:
    """
    Run beam search on the draft model using FlashInfer paged attention.

    Args:
        model: HF model (with FlashInfer-patched attention)
        request_kv_cache: Pre-allocated request KV cache
        input_ids: Prompt token IDs [1, seq_len]
        config: BeamSearchConfig with beam width, depth, etc.
        flashinfer_wrapper: FlashInfer attention wrapper
        prefilled_logits: If provided, skip draft prefill and use these logits
            directly. The request_kv_cache must already contain the KV data
            (e.g., copied from the target model's prefill). Shape: [1, seq_len, vocab].

    Returns:
        tree: Draft tree containing all beam search candidates
        step_times: List of per-step decode times in seconds
        cuda_graph_runner: The CudaGraphRunner (persisted for reuse), or None
    """
    device = input_ids.device
    dtype = model.lm_head.weight.dtype
    batch_size, input_len = input_ids.shape

    K = config.topk_len
    max_depth = config.max_depth
    kvCachePool = request_kv_cache.kvCachePool
    PAGE_SIZE = kvCachePool.page_len
    # --- KV length init ---
    kv_len = request_kv_cache.get_seq_length()
    if isinstance(kv_len, torch.Tensor):
        kv_len = kv_len.item()

    # --- Prefill (or skip if using shared KV) ---
    t_prefill_start = time.perf_counter()

    if prefilled_logits is not None:
        # KV cache already populated (shared from target model).
        # kv_len already accounts for prompt tokens.
        logits = prefilled_logits
        org_kv_len = kv_len
    else:
        # Draft model does its own prefill.
        request_kv_cache.increment(input_len)
        batch_position = getKvCacheBatchPosition(
            request_kv_caches=[request_kv_cache],
            mode="tree",
            device=device,
            treeTokens=input_len,
        )
        flashinfer_wrapper.prepareAttention(
            "prefill", batch_position, PAGE_SIZE, "NONE", kvCachePool.cache_data[0].dtype,
        )
        position_ids = torch.arange(kv_len, kv_len + input_len, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model(
            input_ids,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            kvCachePool=kvCachePool,
            batch_position=batch_position,
            mode="prefill",
            flashinferWrapper=flashinfer_wrapper,
        )
        logits = outputs.logits
        kv_len += input_len
        org_kv_len = kv_len

    # CUDA graph setup
    cuda_runner = None
    if config.use_cuda_graph:
        if cuda_graph_runner is not None and cuda_graph_runner.graph is not None:
            # Reuse existing runner — do NOT reinit decode wrapper (graph holds
            # references to the old wrapper's internal buffers).
            cuda_runner = cuda_graph_runner
        else:
            flashinfer_wrapper.init_cuda_graph_decode(K, kvCachePool.max_pages, device)
            if cuda_graph_runner is not None:
                cuda_runner = cuda_graph_runner
            else:
                cuda_runner = CudaGraphRunner(K, kvCachePool.max_pages, device)

    step_times = [time.perf_counter() - t_prefill_start]

    # --- Init tree ---
    tree = Tree(input_ids[0, -1], dtype)
    prompt_pages = list(request_kv_cache.kv_page_indices)

    # --- Get K initial tokens from prefill output ---
    sampled_probs = torch.softmax(logits[0, -1, :] / config.temperature, dim=-1)
    topk_probs, topk_ids = sampled_probs.topk(K)

    # --- Init trie state ---
    node_pages: dict[int, list[int]] = {}
    page_ref_counts: dict[int, int] = {}

    beam_node: list[int] = []
    for i in range(K):
        tok = topk_ids[i].item()
        prob = topk_probs[i].item()
        new_idx = tree.current_size
        tn = TreeNode(parent=0, token_id=tok, cumulative_probability=prob, depth=1)
        tree.nodes[0].children.append(new_idx)
        tree.nodes.append(tn)
        tree.current_size += 1
        node_pages[new_idx] = list(prompt_pages)
        for p in prompt_pages:
            page_ref_counts[p] = page_ref_counts.get(p, 0) + 1
        beam_node.append(new_idx)

    tree.available_leaves = list(beam_node)
    cum_log_probs = torch.log(topk_probs).tolist()

    # --- Decode loop ---
    current_pos = kv_len
    for step in range(max_depth - 1):
        t_step_start = time.perf_counter()
        off = current_pos % PAGE_SIZE
        pli = current_pos // PAGE_SIZE

        # COW enforcement
        for node_idx in set(beam_node):
            if off == 0:
                new_page = kvCachePool.allocate(1)[0]
                node_pages[node_idx].append(new_page)
                page_ref_counts[new_page] = 1
            else:
                write_page = node_pages[node_idx][pli]
                if page_ref_counts[write_page] > 1:
                    new_page = _copy_block(kvCachePool, write_page, off)
                    page_ref_counts[write_page] -= 1
                    if page_ref_counts[write_page] == 0:
                        kvCachePool.deallocate([write_page])
                        del page_ref_counts[write_page]
                    node_pages[node_idx][pli] = new_page
                    page_ref_counts[new_page] = 1

        # Build batch position for K beams
        beam_pages_list = [node_pages[beam_node[k]] for k in range(K)]
        batch_position = _build_beam_batch_position(beam_pages_list, current_pos, PAGE_SIZE, device)

        # Batched decode forward: [K, 1]
        beam_input_ids = torch.tensor(
            [[tree.nodes[beam_node[k]].token_id] for k in range(K)],
            dtype=torch.long, device=device,
        )
        beam_position_ids = torch.full((K, 1), current_pos, dtype=torch.long, device=device)

        if cuda_runner is not None and cuda_runner.graph is not None:
            # CUDA graph replay path
            outputs = cuda_runner.replay(
                beam_input_ids, beam_position_ids, batch_position,
                flashinfer_wrapper, PAGE_SIZE, dtype,
            )
        else:
            flashinfer_wrapper.prepareAttention(
                "decode", batch_position, PAGE_SIZE, "NONE", kvCachePool.cache_data[0].dtype,
            )
            outputs = model(
                beam_input_ids,
                position_ids=beam_position_ids,
                past_key_values=None,
                use_cache=False,
                kvCachePool=kvCachePool,
                batch_position=batch_position,
                mode="decode",
                flashinferWrapper=flashinfer_wrapper,
            )
            # Capture graph after first successful decode forward
            if cuda_runner is not None and cuda_runner.graph is None:
                cuda_runner.capture(
                    model, flashinfer_wrapper, kvCachePool, PAGE_SIZE,
                    dtype,
                )

        logits = outputs.logits

        # Score and select top-K
        probs = torch.softmax(logits[:, -1, :] / config.temperature, dim=-1)
        cum = torch.tensor(cum_log_probs, device=device)
        flat_scores = (cum[:, None] + torch.log(probs + 1e-10)).reshape(-1)
        topk_scores, topk_flat_ids = flat_scores.topk(K)
        vocab_size = probs.shape[-1]
        parent_list = (topk_flat_ids // vocab_size).tolist()
        new_tok_list = (topk_flat_ids % vocab_size).tolist()

        # Build new trie nodes with dedup
        seen_pairs: dict[tuple, int] = {}
        new_beam_node: list[int] = []
        for i in range(K):
            parent_node = beam_node[parent_list[i]]
            tok = new_tok_list[i]
            key = (parent_node, tok)
            step_prob = probs[parent_list[i], tok].item()

            if key in seen_pairs:
                new_node = seen_pairs[key]
            else:
                existing = tree.find_child_index(parent_node, tok)
                if existing != -1 and existing in node_pages:
                    new_node = existing
                else:
                    new_node = tree.current_size
                    tn = TreeNode(
                        parent=parent_node,
                        token_id=tok,
                        cumulative_probability=step_prob,
                        depth=tree.nodes[parent_node].depth + 1,
                    )
                    tree.nodes[parent_node].children.append(new_node)
                    tree.nodes.append(tn)
                    tree.current_size += 1
                    node_pages[new_node] = list(node_pages[parent_node])
                    for p in node_pages[new_node]:
                        page_ref_counts[p] += 1
                seen_pairs[key] = new_node
            new_beam_node.append(new_node)

        # Release old live nodes
        old_unique = set(beam_node)
        for old_node in old_unique:
            pages_to_free = []
            for p in node_pages[old_node]:
                page_ref_counts[p] -= 1
                if page_ref_counts[p] == 0:
                    pages_to_free.append(p)
                    del page_ref_counts[p]
            for p in pages_to_free:
                kvCachePool.deallocate([p])
            del node_pages[old_node]

        tree.available_leaves = list(set(new_beam_node))
        beam_node = new_beam_node
        cum_log_probs = topk_scores.tolist()
        current_pos += 1

        step_times.append(time.perf_counter() - t_step_start)

    # --- Cleanup: free all beam-allocated (non-prompt) pages ---
    prompt_pages_set = set(prompt_pages)
    pages_to_free: set[int] = set()
    for pages in node_pages.values():
        for p in pages:
            if p not in prompt_pages_set:
                pages_to_free.add(p)
    for p in pages_to_free:
        kvCachePool.deallocate([p])

    return tree, step_times, cuda_runner
