"""Profile model.forward() for draft (GemLite) vs target (bf16) at batch=1."""

import sys, time, torch, logging
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO)

from parallel_tree_spec.experiment import BeamSearchExperiment, DEFAULT_PROMPTS
from parallel_tree_spec.flashinfer.cache_manager import (
    RequestKvCache, getKvCacheBatchPosition, copy_kv_pages,
)

device = torch.device("cuda")

exp = BeamSearchExperiment(beam_width=6, max_depth=10, share_kv=True, temperature=0.2)
exp.load_target_model()
exp.load_draft_model(nbits=4, group_size=64)

prompt = DEFAULT_PROMPTS[0]
prompt_ids = exp.tokenizer.encode(prompt, return_tensors="pt").to(device)

# --- Prefill target ---
exp.target_kv_pool.reset()
exp.draft_kv_pool.reset()

target_kv = RequestKvCache(kvCachePool=exp.target_kv_pool, page_len=exp.page_len, seq_init_len=0)
target_kv.increment(prompt_ids.shape[1])
bp = getKvCacheBatchPosition([target_kv], mode="tree", device=device, treeTokens=prompt_ids.shape[1])
exp.target_wrapper.prepareAttention("prefill", bp, exp.page_len, "NONE", exp.target_kv_pool.cache_data[0].dtype)
pos = torch.arange(prompt_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
target_out = exp.target_model(
    prompt_ids, position_ids=pos, past_key_values=None, use_cache=False,
    kvCachePool=exp.target_kv_pool, batch_position=bp, mode="prefill",
    flashinferWrapper=exp.target_wrapper,
)

# --- Setup draft KV ---
draft_kv = copy_kv_pages(exp.target_kv_pool, target_kv, exp.draft_kv_pool)

seq_len = prompt_ids.shape[1]

def bench_decode(model, wrapper, kv_pool, kv_cache, label, batch_sizes=[1, 6]):
    for bs in batch_sizes:
        # Build batch position for bs tokens at position seq_len
        kv_cache_copy_pages = list(kv_cache.kv_page_indices)
        beam_pages_list = [kv_cache_copy_pages for _ in range(bs)]

        from parallel_tree_spec.beam_search import _build_beam_batch_position
        batch_pos = _build_beam_batch_position(beam_pages_list, seq_len, exp.page_len, device)

        input_ids = torch.zeros((bs, 1), dtype=torch.long, device=device)
        position_ids = torch.full((bs, 1), seq_len, dtype=torch.long, device=device)

        wrapper.prepareAttention("decode", batch_pos, exp.page_len, "NONE", kv_pool.cache_data[0].dtype)

        # Warmup
        for _ in range(3):
            model(input_ids, position_ids=position_ids, past_key_values=None, use_cache=False,
                  kvCachePool=kv_pool, batch_position=batch_pos, mode="decode",
                  flashinferWrapper=wrapper)

        # Benchmark
        torch.cuda.synchronize()
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            model(input_ids, position_ids=position_ids, past_key_values=None, use_cache=False,
                  kvCachePool=kv_pool, batch_position=batch_pos, mode="decode",
                  flashinferWrapper=wrapper)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / N * 1000
        print(f"  {label} batch={bs}: {elapsed:.2f} ms")

print("\n" + "="*60)
print("MODEL FORWARD BENCHMARK (decode, no attention plan overhead)")
print("="*60)

bench_decode(exp.target_model, exp.target_wrapper, exp.target_kv_pool, target_kv, "target(bf16)")
bench_decode(exp.draft_model, exp.draft_wrapper, exp.draft_kv_pool, draft_kv, "draft(4b/g64)")

# Also benchmark just the linear layers
print("\n" + "="*60)
print("TORCH PROFILER: draft forward kernel breakdown")
print("="*60)

from parallel_tree_spec.beam_search import _build_beam_batch_position
batch_pos = _build_beam_batch_position([list(draft_kv.kv_page_indices)], seq_len, exp.page_len, device)
input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
position_ids = torch.full((1, 1), seq_len, dtype=torch.long, device=device)
exp.draft_wrapper.prepareAttention("decode", batch_pos, exp.page_len, "NONE", exp.draft_kv_pool.cache_data[0].dtype)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(3):
        exp.draft_model(input_ids, position_ids=position_ids, past_key_values=None, use_cache=False,
                        kvCachePool=exp.draft_kv_pool, batch_position=batch_pos, mode="decode",
                        flashinferWrapper=exp.draft_wrapper)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Same for target
print("\n" + "="*60)
print("TORCH PROFILER: target forward kernel breakdown")
print("="*60)

batch_pos_t = _build_beam_batch_position([list(target_kv.kv_page_indices)], seq_len, exp.page_len, device)
exp.target_wrapper.prepareAttention("decode", batch_pos_t, exp.page_len, "NONE", exp.target_kv_pool.cache_data[0].dtype)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof2:
    for _ in range(3):
        exp.target_model(input_ids, position_ids=position_ids, past_key_values=None, use_cache=False,
                         kvCachePool=exp.target_kv_pool, batch_position=batch_pos_t, mode="decode",
                         flashinferWrapper=exp.target_wrapper)
        torch.cuda.synchronize()

print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=20))
