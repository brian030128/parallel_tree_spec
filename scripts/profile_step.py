"""Profile beam search step to find per-component time breakdown."""

import sys, time, torch, logging, argparse
sys.path.insert(0, "src")

logging.basicConfig(level=logging.INFO)

from parallel_tree_spec.experiment import BeamSearchExperiment, DEFAULT_PROMPTS
from parallel_tree_spec.beam_search import (
    _build_beam_batch_position, _build_cascade_data, _copy_block,
    BeamSearchConfig,
)
from parallel_tree_spec.flashinfer.cache_manager import (
    KvCacheBatchPosition, KvCachePool, RequestKvCache, getKvCacheBatchPosition,
    copy_kv_pages,
)
from parallel_tree_spec.tree import Tree, TreeNode


@torch.no_grad()
def profiled_beam_search(
    model, request_kv_cache, input_ids, config, flashinfer_wrapper,
    prefilled_logits=None,
):
    """Beam search with per-component timing."""
    device = input_ids.device
    dtype = model.lm_head.weight.dtype
    K = config.topk_len
    max_depth = config.max_depth
    kvCachePool = request_kv_cache.kvCachePool
    PAGE_SIZE = kvCachePool.page_len
    kv_len = request_kv_cache.get_seq_length()
    if isinstance(kv_len, torch.Tensor):
        kv_len = kv_len.item()

    logits = prefilled_logits
    org_kv_len = kv_len

    num_shared_pages = org_kv_len // PAGE_SIZE
    use_cascade = config.use_cascade

    if use_cascade and num_shared_pages > 0 and flashinfer_wrapper.cascade_wrapper is None:
        flashinfer_wrapper.init_cascade_decode(2)

    # Init tree + beams
    tree = Tree(input_ids[0, -1], dtype)
    prompt_pages = list(request_kv_cache.kv_page_indices)

    sampled_probs = torch.softmax(logits[0, -1, :] / config.temperature, dim=-1)
    topk_probs, topk_ids = sampled_probs.topk(K)

    node_pages = {}
    page_ref_counts = {}
    beam_node = []
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

    cum_log_probs = torch.log(topk_probs).tolist()

    timings = {
        "cow": [], "build_batch": [], "build_cascade": [],
        "prepare_cascade": [], "prepare_decode": [],
        "forward": [], "scoring": [], "tree_update": [],
        "total": [],
    }

    current_pos = kv_len
    for step in range(max_depth - 1):
        torch.cuda.synchronize()
        t_total = time.perf_counter()

        # --- COW ---
        t0 = time.perf_counter()
        off = current_pos % PAGE_SIZE
        pli = current_pos // PAGE_SIZE
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
        timings["cow"].append(time.perf_counter() - t0)

        # --- Build batch position ---
        t0 = time.perf_counter()
        beam_pages_list = [node_pages[beam_node[k]] for k in range(K)]
        batch_position = _build_beam_batch_position(beam_pages_list, current_pos, PAGE_SIZE, device)
        timings["build_batch"].append(time.perf_counter() - t0)

        # --- Build inputs ---
        beam_input_ids = torch.tensor(
            [[tree.nodes[beam_node[k]].token_id] for k in range(K)],
            dtype=torch.long, device=device,
        )
        beam_position_ids = torch.full((K, 1), current_pos, dtype=torch.long, device=device)

        # --- Attention prep + forward ---
        if num_shared_pages > 0:
            t0 = time.perf_counter()
            cascade_data = _build_cascade_data(
                beam_pages_list, num_shared_pages, current_pos, PAGE_SIZE, device
            )
            timings["build_cascade"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            flashinfer_wrapper.prepareCascadeAttention(
                cascade_data.qo_indptr_arr,
                cascade_data.kv_page_indptr_arr,
                cascade_data.kv_page_indices_arr,
                cascade_data.kv_last_page_len_arr,
                PAGE_SIZE,
                kvCachePool.cache_data[0].dtype,
            )
            timings["prepare_cascade"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            flashinfer_wrapper.prepareAttention(
                "decode", batch_position, PAGE_SIZE, "NONE", kvCachePool.cache_data[0].dtype,
            )
            timings["prepare_decode"].append(time.perf_counter() - t0)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(
                beam_input_ids, position_ids=beam_position_ids,
                past_key_values=None, use_cache=False,
                kvCachePool=kvCachePool, batch_position=batch_position,
                mode="cascade_decode", flashinferWrapper=flashinfer_wrapper,
            )
            torch.cuda.synchronize()
            timings["forward"].append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            flashinfer_wrapper.prepareAttention(
                "decode", batch_position, PAGE_SIZE, "NONE", kvCachePool.cache_data[0].dtype,
            )
            timings["prepare_decode"].append(time.perf_counter() - t0)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(
                beam_input_ids, position_ids=beam_position_ids,
                past_key_values=None, use_cache=False,
                kvCachePool=kvCachePool, batch_position=batch_position,
                mode="decode", flashinferWrapper=flashinfer_wrapper,
            )
            torch.cuda.synchronize()
            timings["forward"].append(time.perf_counter() - t0)

        logits_out = outputs.logits

        # --- Scoring ---
        t0 = time.perf_counter()
        probs = torch.softmax(logits_out[:, -1, :] / config.temperature, dim=-1)
        cum = torch.tensor(cum_log_probs, device=device)
        flat_scores = (cum[:, None] + torch.log(probs + 1e-10)).reshape(-1)
        topk_scores, topk_flat_ids = flat_scores.topk(K)
        vocab_size = probs.shape[-1]
        parent_list = (topk_flat_ids // vocab_size).tolist()
        new_tok_list = (topk_flat_ids % vocab_size).tolist()
        timings["scoring"].append(time.perf_counter() - t0)

        # --- Tree update ---
        t0 = time.perf_counter()
        seen_pairs = {}
        new_beam_node = []
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
                    tn = TreeNode(parent=parent_node, token_id=tok,
                                  cumulative_probability=step_prob,
                                  depth=tree.nodes[parent_node].depth + 1)
                    tree.nodes[parent_node].children.append(new_node)
                    tree.nodes.append(tn)
                    tree.current_size += 1
                    node_pages[new_node] = list(node_pages[parent_node])
                    for p in node_pages[new_node]:
                        page_ref_counts[p] += 1
                seen_pairs[key] = new_node
            new_beam_node.append(new_node)

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

        beam_node = new_beam_node
        cum_log_probs = topk_scores.tolist()
        current_pos += 1
        timings["tree_update"].append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        timings["total"].append(time.perf_counter() - t_total)

    # Cleanup
    prompt_pages_set = set(prompt_pages)
    pages_to_free_set = set()
    for pages in node_pages.values():
        for p in pages:
            if p not in prompt_pages_set:
                pages_to_free_set.add(p)
    for p in pages_to_free_set:
        kvCachePool.deallocate([p])

    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-width", type=int, default=6)
    args = parser.parse_args()

    K = args.beam_width
    device = torch.device("cuda")

    # Use BeamSearchExperiment to set up models
    exp = BeamSearchExperiment(
        beam_width=K, max_depth=10, share_kv=True, temperature=0.2,
    )
    exp.load_target_model()
    exp.load_draft_model(nbits=4, group_size=64)

    prompt = DEFAULT_PROMPTS[0]
    prompt_ids = exp.tokenizer.encode(prompt, return_tensors="pt").to(device)

    # --- Prefill target ---
    exp.target_kv_pool.reset()
    exp.draft_kv_pool.reset()

    target_kv_cache = RequestKvCache(
        kvCachePool=exp.target_kv_pool, page_len=exp.page_len, seq_init_len=0,
    )
    target_kv_cache.increment(prompt_ids.shape[1])
    target_batch_pos = getKvCacheBatchPosition(
        request_kv_caches=[target_kv_cache], mode="tree",
        device=device, treeTokens=prompt_ids.shape[1],
    )
    exp.target_wrapper.prepareAttention(
        "prefill", target_batch_pos, exp.page_len, "NONE",
        exp.target_kv_pool.cache_data[0].dtype,
    )
    position_ids = torch.arange(prompt_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    target_outputs = exp.target_model(
        prompt_ids, position_ids=position_ids,
        past_key_values=None, use_cache=False,
        kvCachePool=exp.target_kv_pool, batch_position=target_batch_pos,
        mode="prefill", flashinferWrapper=exp.target_wrapper,
    )

    def run_profiled():
        exp.draft_kv_pool.reset()
        draft_kv_cache = copy_kv_pages(
            src_pool=exp.target_kv_pool,
            src_request=target_kv_cache,
            dst_pool=exp.draft_kv_pool,
        )
        config = BeamSearchConfig(
            topk_len=K, max_depth=10, temperature=0.2, use_cascade=True,
        )
        # Reset cascade wrapper so it re-inits
        exp.draft_wrapper.cascade_wrapper = None
        return profiled_beam_search(
            exp.draft_model, draft_kv_cache, prompt_ids, config,
            exp.draft_wrapper, prefilled_logits=target_outputs.logits,
        )

    # Warmup
    run_profiled()

    # Actual profiling run
    timings = run_profiled()

    print(f"\n{'='*60}")
    print(f"BEAM SEARCH STEP PROFILE (K={K})")
    print(f"{'='*60}")
    for name, vals in timings.items():
        if vals:
            avg = sum(vals) / len(vals) * 1000
            print(f"  {name:20s}: {avg:8.3f} ms avg ({len(vals)} steps)")

    print(f"\n{'='*60}")
    print("Per-step detail (step 2 onward, skip warmup):")
    print(f"{'='*60}")
    for step in range(1, len(timings["total"])):
        print(f"\n  Step {step+1}:")
        for name, vals in timings.items():
            if step < len(vals):
                print(f"    {name:20s}: {vals[step]*1000:8.3f} ms")


if __name__ == "__main__":
    main()
