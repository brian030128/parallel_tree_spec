"""
Benchmark: single-token decode latency of HQQ draft model vs bf16 target model.

Verifies that GemLite backend is active and that quantized draft model
achieves lower per-token decode latency than the full-precision target.

Usage:
    cd parallel_tree_spec
    uv run python scripts/bench_decode_latency.py
"""

from __future__ import annotations

import logging
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parallel_tree_spec.flashinfer.attention_wrapper import BeFlashinferWrapper
from parallel_tree_spec.flashinfer.cache_manager import (
    KvCachePool,
    RequestKvCache,
    getKvCacheBatchPosition,
)
from parallel_tree_spec.flashinfer.monkey_patch import apply_flashinfer_kernel_to_llama
from parallel_tree_spec.quantization import make_quant_config, quantize_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
PAGE_LEN = 16
MAX_PAGES = 4096
NBITS = 4
GROUP_SIZE = 64
NUM_WARMUP = 10
NUM_ITERS = 100
PROMPT = "The theory of general relativity, proposed by Albert Einstein in 1915,"


def _extract_model_dims(model):
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return num_layers, cfg.num_attention_heads, num_kv_heads, head_dim, cfg.hidden_size


def _setup_kv(model, dtype):
    """Create KV cache pool and FlashInfer wrapper for a model."""
    num_layers, num_heads, num_kv_heads, head_dim, hidden_size = _extract_model_dims(model)
    pool = KvCachePool(
        max_pages=MAX_PAGES,
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        page_len=PAGE_LEN,
        dtype=dtype,
        device=DEVICE,
    )
    wrapper = BeFlashinferWrapper(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=hidden_size,
        page_len=PAGE_LEN,
    )
    return pool, wrapper


def _prefill(model, pool, wrapper, prompt_ids):
    """Run prefill, return (logits, kv_cache)."""
    pool.reset()
    kv = RequestKvCache(kvCachePool=pool, page_len=PAGE_LEN, seq_init_len=0)
    seq_len = prompt_ids.shape[1]
    kv.increment(seq_len)

    batch_pos = getKvCacheBatchPosition(
        request_kv_caches=[kv], mode="tree", device=DEVICE, treeTokens=seq_len,
    )
    wrapper.prepareAttention(
        "prefill", batch_pos, PAGE_LEN, "NONE", pool.cache_data[0].dtype,
    )
    position_ids = torch.arange(seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out = model(
            prompt_ids,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            kvCachePool=pool,
            batch_position=batch_pos,
            mode="prefill",
            flashinferWrapper=wrapper,
        )
    return out.logits, kv


def _bench_decode(model, pool, wrapper, kv, prompt_len, first_token_id, num_warmup, num_iters):
    """Benchmark single-token decode. Returns list of per-iteration times (seconds)."""
    inp = torch.tensor([[first_token_id]], dtype=torch.long, device=DEVICE)
    pos = torch.tensor([[prompt_len]], dtype=torch.long, device=DEVICE)

    times = []
    for i in range(num_warmup + num_iters):
        kv.increment(1)
        batch_pos = getKvCacheBatchPosition(
            request_kv_caches=[kv], mode="tree", device=DEVICE, treeTokens=1,
        )
        wrapper.prepareAttention(
            "decode", batch_pos, PAGE_LEN, "NONE", pool.cache_data[0].dtype,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(
                inp,
                position_ids=pos,
                past_key_values=None,
                use_cache=False,
                kvCachePool=pool,
                batch_position=batch_pos,
                mode="decode",
                flashinferWrapper=wrapper,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= num_warmup:
            times.append(t1 - t0)

        kv.decrement(1)

    return times


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    prompt_len = prompt_ids.shape[1]
    logger.info(f"Prompt: {prompt_len} tokens")

    # ── Target model (bf16) ─────────────────────────────────────────────
    logger.info("Loading target model (bf16)...")
    target = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    target.eval()
    apply_flashinfer_kernel_to_llama(target)
    t_pool, t_wrap = _setup_kv(target, torch.bfloat16)

    logits, t_kv = _prefill(target, t_pool, t_wrap, prompt_ids)
    first_tok = logits[0, -1, :].argmax().item()
    logger.info(f"Target prefill done, first decode token: {first_tok}")

    target_times = _bench_decode(
        target, t_pool, t_wrap, t_kv, prompt_len, first_tok, NUM_WARMUP, NUM_ITERS,
    )
    t_kv.release()

    # Free target to make room for draft
    del target, t_pool, t_wrap, t_kv
    torch.cuda.empty_cache()

    # ── Draft model (HQQ 4b/g64) ───────────────────────────────────────
    logger.info(f"Loading draft model (HQQ {NBITS}b/g{GROUP_SIZE})...")
    draft = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    draft.eval()
    apply_flashinfer_kernel_to_llama(draft)
    qcfg = make_quant_config(draft, nbits=NBITS, group_size=GROUP_SIZE)
    quantize_model(draft, qcfg, compute_dtype=torch.float16, device=DEVICE)
    draft.eval()  # GemLite layers default to training=True

    d_pool, d_wrap = _setup_kv(draft, torch.float16)

    logits_d, d_kv = _prefill(draft, d_pool, d_wrap, prompt_ids)
    draft_first_tok = logits_d[0, -1, :].argmax().item()
    logger.info(f"Draft prefill done, first decode token: {draft_first_tok}")

    draft_times = _bench_decode(
        draft, d_pool, d_wrap, d_kv, prompt_len, draft_first_tok, NUM_WARMUP, NUM_ITERS,
    )
    d_kv.release()

    # ── Results ─────────────────────────────────────────────────────────
    def _stats(times):
        return {
            "median_ms": statistics.median(times) * 1000,
            "mean_ms": statistics.mean(times) * 1000,
            "stdev_ms": statistics.stdev(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }

    ts = _stats(target_times)
    ds = _stats(draft_times)
    speedup = ts["median_ms"] / ds["median_ms"]

    print("\n" + "=" * 65)
    print(f"  Single-token decode latency  ({NUM_ITERS} iters, {NUM_WARMUP} warmup)")
    print("=" * 65)
    print(f"{'':>20s}  {'Target (bf16)':>14s}  {'Draft (HQQ)':>14s}")
    print(f"{'':>20s}  {'─' * 14}  {'─' * 14}")
    print(f"{'Median':>20s}  {ts['median_ms']:>11.3f} ms  {ds['median_ms']:>11.3f} ms")
    print(f"{'Mean':>20s}  {ts['mean_ms']:>11.3f} ms  {ds['mean_ms']:>11.3f} ms")
    print(f"{'Stdev':>20s}  {ts['stdev_ms']:>11.3f} ms  {ds['stdev_ms']:>11.3f} ms")
    print(f"{'Min':>20s}  {ts['min_ms']:>11.3f} ms  {ds['min_ms']:>11.3f} ms")
    print(f"{'Max':>20s}  {ts['max_ms']:>11.3f} ms  {ds['max_ms']:>11.3f} ms")
    print(f"\n  Speedup (target / draft median): {speedup:.2f}x")
    if speedup > 1.0:
        print("  ✓ Draft model is FASTER than target")
    else:
        print("  ✗ Draft model is SLOWER — GemLite may not be active")
    print("=" * 65)


if __name__ == "__main__":
    main()
