# FlashInfer Paged Attention for Beam Search

## Why FlashInfer?

Standard HuggingFace beam search duplicates the entire KV cache for each beam, using O(K * seq_len) memory. FlashInfer's **paged KV cache** with **copy-on-write (COW)** allows beams to share prompt pages, reducing memory to O(prompt_len + K * beam_depth).

## Paged KV Cache Architecture

### KvCachePool

A pool of fixed-size pages, allocated as a single GPU tensor:

```python
cache_data shape: [num_layers, max_pages, 2, page_len, num_heads, head_dim]
                   │           │          │   │         │          │
                   │           │          │   │         │          └─ head dimension (e.g. 128)
                   │           │          │   │         └─ KV heads (e.g. 8 for GQA)
                   │           │          │   └─ tokens per page (e.g. 16)
                   │           │          └─ 0=keys, 1=values
                   │           └─ total page slots
                   └─ transformer layers
```

Pages are allocated/freed independently:
- `allocate(n)` → returns n free page indices
- `deallocate(pages)` → marks pages as free
- Tracked via a `free_page_mask` boolean array

### RequestKvCache

Per-request wrapper that manages an ordered list of page indices:
- `kv_page_indices: list[int]` — physical pages for this request's KV sequence
- `increment(n)` — grows the sequence, allocates new pages when current page is full
- `release()` — returns all pages to the pool

### KvCacheBatchPosition

Describes a batch of sequences for FlashInfer kernels:
```python
seq_indptr        # [batch+1] cumulative token counts per sequence
kv_page_indptr    # [batch+1] cumulative page counts per sequence
kv_page_indices   # [total_pages] physical page IDs
kv_last_page_len  # [batch] filled slots in each sequence's last page
batch_indices     # [total_tokens] which batch item each token belongs to
positions         # [total_tokens] ABSOLUTE sequence positions (0, 1, 2, ...)
```

## FlashInfer Wrappers

### BeFlashinferWrapper

Wraps two core FlashInfer wrappers:
1. `BatchPrefillWithPagedKVCacheWrapper` — processes multiple tokens per sequence (prefill, tree attention)
2. `BatchDecodeWithPagedKVCacheWrapper` — processes exactly 1 token per sequence (decode)

Key methods:

**`prepareAttention(mode, batch_position, ...)`** — calls `.plan()` to configure the kernel:
- `mode="prefill"`: uses prefill wrapper with `causal=True`
- `mode="decode"`: uses decode wrapper
- `mode="tree"`: uses prefill wrapper with `custom_mask=attention_mask, causal=False` (tree attention)

**`computeAttention(q, k, v, cacheData, mode, ...)`** — runs attention:
1. Appends new K,V to the paged cache at correct positions
2. Runs the FlashInfer attention kernel (prefill or decode)
3. Returns attention output

### Cascade Attention (Optional Optimization)

For beam search, all beams share the same prompt KV cache. Cascade attention splits computation into:
- **Level 0 (shared)**: prompt pages, computed once for all K beams
- **Level 1 (unique)**: per-beam decode pages

Uses `flashinfer.MultiLevelCascadeAttentionWrapper` with 2 levels.

## Beam Search with Paged KV Cache

### Phase 1: Prefill

```python
# Allocate pages for prompt
request_kv_cache.increment(prompt_len)

# Forward prompt through FlashInfer prefill wrapper
batch_position = getKvCacheBatchPosition(mode='prefill')
flashinferWrapper.prepareAttention('prefill', batch_position, ...)
logits = model(prompt_ids, mode='prefill', ...)

# Initialize K beams from top-K tokens
topk_probs, topk_ids = logits[-1].topk(K)
prompt_pages = list(request_kv_cache.kv_page_indices)
```

### Phase 2: Decode Loop (Copy-on-Write Beam Search)

Each beam maintains its own list of page indices (`node_pages[node_idx]`). Reference counting tracks shared pages.

```python
for step in range(max_depth - 1):
    off = current_pos % PAGE_SIZE   # within-page offset
    pli = current_pos // PAGE_SIZE  # page list index

    # COW enforcement: ensure each beam has a unique write page
    for node_idx in set(beam_node):
        if off == 0:
            # New page boundary — allocate fresh page
            new_page = kvCachePool.allocate(1)[0]
            node_pages[node_idx].append(new_page)
        else:
            write_page = node_pages[node_idx][pli]
            if page_ref_counts[write_page] > 1:
                # Shared page — copy first `off` slots to new page (COW)
                new_page = _copy_block(kvCachePool, write_page, off)
                page_ref_counts[write_page] -= 1
                node_pages[node_idx][pli] = new_page
                page_ref_counts[new_page] = 1

    # Build batch position for K beams
    beam_pages_list = [node_pages[beam_node[k]] for k in range(K)]
    batch_position = _build_beam_batch_position(beam_pages_list, current_pos, PAGE_SIZE)

    # Batched decode forward: [K, 1] input
    beam_input_ids = [[tree.nodes[beam_node[k]].token_id] for k in range(K)]
    logits = model(beam_input_ids, mode='decode', batch_position=batch_position, ...)

    # Score and select top-K
    probs = softmax(logits / temperature)
    flat_scores = cum_log_probs[:, None] + log(probs)  # [K, vocab]
    topk_scores, topk_flat = flat_scores.flatten().topk(K)
    parent_list = topk_flat // vocab_size
    new_tok_list = topk_flat % vocab_size

    # Build new tree nodes with trie dedup
    for i in range(K):
        parent_node = beam_node[parent_list[i]]
        tok = new_tok_list[i]
        # Create new TreeNode, inherit parent's pages, update ref counts
        node_pages[new_node] = list(node_pages[parent_node])
        for p in node_pages[new_node]:
            page_ref_counts[p] += 1

    # Release old beams' page refs
    for old_node in set(beam_node):
        for p in node_pages[old_node]:
            page_ref_counts[p] -= 1
            if page_ref_counts[p] == 0:
                kvCachePool.deallocate([p])

    current_pos += 1
```

### Phase 3: Cleanup

```python
# Free all beam-allocated (non-prompt) pages
prompt_pages_set = set(prompt_pages)
for pages in node_pages.values():
    for p in pages:
        if p not in prompt_pages_set:
            pages_to_free.add(p)
kvCachePool.deallocate(list(pages_to_free))
```

## Memory Comparison

For Llama-3.1-8B (32 layers, 8 KV heads, 128 head_dim, fp16):

| Method | Memory for K=6, depth=10, prompt=512 |
|--------|--------------------------------------|
| HF naive (copy full cache per beam) | 6 × 512 × 32 × 8 × 128 × 2 × 2B = ~6GB |
| FlashInfer paged (shared prompt) | 512 + 6×10 tokens ≈ 572 tokens worth = ~0.6GB |

## CUDA Graph Support

The beam decode loop can be captured as a CUDA graph:
1. `init_cuda_graph_decode()` — pre-allocates staging buffers, captures forward pass
2. `beam_decode_step()` — copies live data into staging buffers, replays graph
3. **Constraint**: FlashInfer's `.plan()` must be called OUTSIDE the graph; only `.run()` is captured inside

## FlashInfer Monkey Patching

To use FlashInfer with HuggingFace models, we monkey-patch the attention layers:

```python
apply_flashinfer_kernel_to_llama(model, attention=True, rms_norm=True)
```

This replaces:
- `LlamaAttention.forward` → `FiLlamaAttention.forward` (routes Q,K,V through FlashInfer wrappers)
- `LlamaRMSNorm.forward` → Fused RMS norm (optional, for speed)

The patched forward accepts extra kwargs: `flashinferWrapper`, `kvCachePool`, `batch_position`, `mode`.

## Reference Files

- Beam search with COW: `subspec_v2/specdecodes/models/draft_models/be_classic_sd_fi.py`
- FlashInfer wrapper: `subspec_v2/specdecodes/models/utils/flashinfer/be_attention_wrapper.py`
- KV cache manager: `subspec_v2/specdecodes/models/utils/flashinfer/cache_manager.py`
- Attention layer: `subspec_v2/specdecodes/models/utils/flashinfer/attention.py`
- Monkey patch: `subspec_v2/specdecodes/models/utils/flashinfer/monkey_patch.py`
- beam_engine reference: `beam_engine/tests/test_beam_search.py` (simpler standalone version)
