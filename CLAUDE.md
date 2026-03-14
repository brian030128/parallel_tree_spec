# parallel_tree_spec

Evaluate HQQ-quantized Llama-3.1-8B as a draft model in speculative decoding beam search. Self-contained project — all code rewritten from subspec_v2 and beam_engine.

## Status

- [x] Step 1: Project setup (pyproject.toml, directory structure)
- [x] Documentation (docs/)
- [x] Step 2: FlashInfer infrastructure (flashinfer/)
- [x] Step 3: Tree data structure (tree.py)
- [x] Step 4: Beam search draft model (beam_search.py)
- [x] Step 5: Verification (verification.py)
- [x] Step 6: HQQ quantization helper (quantization.py)
- [x] Step 7: Experiment runner (experiment.py)
- [x] Step 8: Metrics & CLI (metrics.py, scripts/run_experiment.py)
- [x] Step 9: Sparse attention (sparse_attention/ package)

All implementation complete. Ready for testing on GPU.

## Project Structure

```
parallel_tree_spec/
├── pyproject.toml
├── CLAUDE.md
├── docs/
│   ├── speculative_decoding.md
│   ├── flashinfer_beam_search.md
│   └── hqq_quantization.md
├── src/parallel_tree_spec/
│   ├── __init__.py
│   ├── flashinfer/
│   │   ├── __init__.py
│   │   ├── attention_wrapper.py    # BeFlashinferWrapper
│   │   ├── cache_manager.py        # KvCachePool, RequestKvCache, KvCacheBatchPosition
│   │   ├── attention.py            # FiLlamaAttention
│   │   └── monkey_patch.py         # apply_flashinfer_kernel_to_llama()
│   ├── sparse_attention/
│   │   ├── __init__.py             # Factory: create_strategy()
│   │   ├── base.py                 # ABC + SparseAttentionConfig
│   │   ├── noop.py                 # NoopStrategy (full attention baseline)
│   │   └── pillar.py               # PillarStrategy (Top-K page selection)
│   ├── tree.py                     # Tree/TreeNode
│   ├── beam_search.py              # COW paged KV beam search
│   ├── verification.py             # verify_tree + target tree decoding
│   ├── quantization.py             # HQQ helpers
│   ├── experiment.py               # Experiment runner
│   └── metrics.py                  # Metrics collection
└── scripts/
    └── run_experiment.py           # CLI entry point
```

## Running

```bash
conda activate flashtree
cd parallel_tree_spec
uv sync
uv run python scripts/run_experiment.py \
    --model meta-llama/Llama-3.1-8B \
    --beam-width 6 --max-depth 10 \
    --quant-configs "4,64;3,64;2,64" \
    --num-prompts 20
```

## Key Design Decisions

1. **FlashInfer for both draft and target**: Efficient paged attention, not HF's standard SDPA
2. **HQQ + GemLite**: Quantized weights, fused dequant+GEMM via GemLite backend
3. **COW beam search**: Beams share prompt KV pages, copy-on-write for diverging paths
4. **Exact verification**: Greedy argmax matching (do_sample=False)
5. **Self-contained**: No imports from subspec_v2 or beam_engine
6. **Pluggable sparse attention**: Page-level KV sparsity during beam decode (pillar strategy), configurable via CLI

## Source Code References

Code is rewritten from these source files:

| Our File | Source |
|----------|--------|
| flashinfer/cache_manager.py | subspec_v2/.../flashinfer/cache_manager.py |
| flashinfer/attention_wrapper.py | subspec_v2/.../flashinfer/be_attention_wrapper.py |
| flashinfer/attention.py | subspec_v2/.../flashinfer/attention.py |
| flashinfer/monkey_patch.py | subspec_v2/.../flashinfer/monkey_patch.py |
| tree.py | subspec_v2/.../utils/cpu_tree.py |
| beam_search.py | subspec_v2/.../draft_models/be_classic_sd_fi.py |
| verification.py | subspec_v2/.../utils/tree_verify.py + generators/classic_sd_fi.py |
| quantization.py | subspec_v2/.../quantizers/hqq/__init__.py |

## Implementation Notes

### FlashInfer Integration Pattern
- `apply_flashinfer_kernel_to_llama(model)` patches LlamaAttention → FiLlamaAttention
- FiLlamaAttention.forward() calls `flashinferWrapper.computeAttention()` which:
  1. Appends K,V to paged cache via `flashinfer.append_paged_kv_cache()`
  2. Runs attention via prefill_wrapper.run() or decode_wrapper.run()
- Model forward must pass extra kwargs: `flashinferWrapper`, `kvCachePool`, `batch_position`, `mode`

### HQQ + FlashInfer Composition
- HQQ quantizes linear weights (QKV, MLP projections)
- Activations remain FP16
- FlashInfer operates on FP16 activations — completely unaware of quantization

### Sparse Attention
- Pluggable via `SparseAttentionStrategy` ABC → `filter_pages()` + `update_importance()`
- **PillarStrategy**: Top-K pages by importance (kv_norm or qk_score), plus recent window
- Page filtering only during decode steps; prefill and tree verification use full attention
- Dual batch positions: sparse for attention `.plan()`, full for `append_kv_cache()`
- CUDA graphs compatible: full pages copied into static buffers for append, `.plan()` called with sparse pages outside the graph
- CLI: `--sparse-method pillar --sparse-budget-ratio 0.05 --sparse-importance kv_norm`

### Beam Search COW Invariant
- `node_pages[node_idx]` = ordered list of physical page indices for node's full KV path
- `page_ref_counts[page]` = number of node_pages lists containing that page
- Before write: if ref > 1, copy page (COW) to get exclusive write access
- On fork: new node inherits parent's pages, increment all refs
- On death: decrement all refs, free pages with ref=0
