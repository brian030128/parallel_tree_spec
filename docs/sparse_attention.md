# Sparse Attention in Beam Search

Page-level KV cache sparsity during beam search decode steps. Only a budget-constrained subset of KV pages are attended to, while all pages still receive new KV writes.

## Why page-level

FlashInfer's paged attention indexes KV cache by page. Filtering at page granularity (typically 16 tokens/page) means we can simply pass a shorter `kv_page_indices` to `.plan()` — no kernel changes, no masking. The cost is coarser granularity than token-level sparsity.

## Architecture

```
SparseAttentionStrategy (ABC)
├── NoopStrategy          # passthrough — returns all pages
└── PillarStrategy        # top-K pages by importance + recent window
```

Factory: `create_strategy(config: SparseAttentionConfig) → SparseAttentionStrategy`

### SparseAttentionConfig fields

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `False` | Master switch |
| `method` | `"none"` | `"none"` or `"pillar"` |
| `budget_ratio` | `0.05` | Fraction of sequence pages to keep |
| `min_budget_tokens` | `128` | Floor on budget (in tokens, not pages) |
| `recent_window` | `32` | Always include last N tokens |
| `importance_method` | `"kv_norm"` | `"kv_norm"` or `"qk_score"` |

### Strategy interface

```python
class SparseAttentionStrategy(ABC):
    def filter_pages(self, all_pages, seq_len, page_len) -> List[int]
    def update_importance(self, kv_cache, page_indices, seq_len, page_len,
                          num_kv_heads, head_dim, q=None)
    def reset()
```

- `update_importance()` — called once after prefill, computes page scores
- `filter_pages()` — called every decode step, returns ordered subset of pages
- `reset()` — called before each new prompt

## PillarStrategy details

### Importance methods

**kv_norm**: Sum of `||K||^2` across all layers/heads for each page. Measures information density — pages with large key norms contribute more to attention.

**qk_score**: `Q @ K^T` using the last prefill token's query vectors (averaged across layers). Directly estimates attention weight each page would receive. Requires an extra pass through `q_proj` + RoPE for the last prompt token (see `_extract_last_token_q` in `beam_search.py`).

### Page selection

1. Compute budget: `max(seq_len * budget_ratio, min_budget_tokens) / page_len`
2. Always include recent window pages (last `recent_window` tokens)
3. Fill remaining budget with highest-importance pages
4. Return selected pages in original order (preserves sequential access pattern)

Cold start: before `update_importance()` runs, `filter_pages()` returns all pages.

## Integration with beam search

### Dual batch positions

Each decode step builds two `KvCacheBatchPosition` objects:

- **full_batch_position**: All pages for every beam. Used for `append_kv_cache` — new KV data must be written to the correct physical page regardless of what attention sees.
- **sparse_batch_position** (= `batch_position` in code when `use_sparse`): Filtered pages. Used for `.plan()` so FlashInfer only reads from the important subset.

In the non-graph path, the model forward receives `batch_position=sparse_batch_position` (for attention) and `append_batch_position=full_batch_position` (for KV append).

### CUDA graph compatibility

CUDA graphs work with sparse attention. The key insight:

- `.plan()` runs **outside** the graph — it accepts any page tensor
- `append_kv_cache` runs **inside** the graph — it reads from `self.batch_position` (static buffers)

At replay time:
1. Copy **full** pages into `self.batch_position` static buffers → graph's append writes correctly
2. Call `.plan()` with **sparse** batch position (a temporary tensor) → attention reads sparse subset
3. Replay graph

```python
cuda_runner.replay(
    beam_input_ids, beam_position_ids,
    full_batch_position,                              # copied into static buffers
    flashinfer_wrapper, PAGE_SIZE, dtype,
    sparse_batch_position=batch_position if use_sparse else None,  # used for .plan()
)
```

No second set of static buffers needed. Expected step time with sparse + CUDA graph: ~12ms (vs ~47ms without graph).

## CLI usage

```bash
uv run python scripts/run_experiment.py \
    --sparse-method pillar \
    --sparse-importance qk_score \
    --sparse-budget-ratio 0.5 \
    --share-kv --temperature 0.2 \
    --prompt-lengths "256,1024"
```

## Gotchas for future work

1. **Importance is static**: Computed once after prefill, never updated during decode. For very long generations this may become stale.

2. **Page granularity**: With 16-token pages, `budget_ratio=0.05` on a 1024-token prompt keeps ~3 pages (48 tokens) + recent window. Very coarse — token-level filtering would be more precise but requires kernel support.

3. **Verification uses full attention**: Sparse filtering only applies to draft beam search decode. Target model prefill/verification always see all KV pages.

4. **`filter_pages` preserves order**: Selected pages are returned in original list order (not sorted by importance). This matters for FlashInfer's sequential page layout expectations.

5. **GQA handling in qk_score**: When `num_q_heads != num_kv_heads`, the Q vectors are reshaped with `repeat_interleave` to match KV head count before computing scores.
