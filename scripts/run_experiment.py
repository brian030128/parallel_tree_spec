#!/usr/bin/env python
"""
CLI entry point for running HQQ-quantized draft model experiments.

Usage:
    cd parallel_tree_spec
    uv run python scripts/run_experiment.py \
        --model meta-llama/Llama-3.1-8B \
        --beam-width 6 --max-depth 10 \
        --quant-configs "4,64;3,64;2,64" \
        --num-prompts 20
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple


def _restrict_visible_gpus():
    """Set CUDA_VISIBLE_DEVICES before any CUDA import to avoid context on unused GPUs."""
    # Parse --device and --draft-device from sys.argv before argparse runs
    gpu_indices = set()
    for flag in ("--device", "--draft-device"):
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                val = sys.argv[idx + 1]
                if val.startswith("cuda:"):
                    gpu_indices.add(val.split(":")[1])
                elif val == "cuda":
                    gpu_indices.add("0")
    if not gpu_indices:
        gpu_indices.add("0")  # default --device is cuda -> cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(gpu_indices))

    # Remap device args to 0-based indices matching the new CUDA_VISIBLE_DEVICES order
    visible = sorted(gpu_indices)
    remap = {orig: str(i) for i, orig in enumerate(visible)}
    for flag in ("--device", "--draft-device"):
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                val = sys.argv[idx + 1]
                if val.startswith("cuda:"):
                    orig_idx = val.split(":")[1]
                    sys.argv[idx + 1] = f"cuda:{remap[orig_idx]}"
                elif val == "cuda":
                    sys.argv[idx + 1] = "cuda:0"


_restrict_visible_gpus()

from parallel_tree_spec.experiment import (
    BeamSearchExperiment,
    DEFAULT_PROMPTS,
    GUTENBERG_DEFAULT_URL,
    download_length_prompts,
)


def parse_quant_configs(config_str: str) -> List[Tuple[int, int]]:
    """Parse quant configs from string like '4,64;3,64;2,64'."""
    configs = []
    for item in config_str.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid quant config: {item!r} (expected 'nbits,group_size')")
        configs.append((int(parts[0]), int(parts[1])))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HQQ-quantized draft models for speculative decoding"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B",
        help="Model name or path (default: meta-llama/Llama-3.1-8B)"
    )
    parser.add_argument(
        "--beam-width", type=int, default=6,
        help="Beam search width K (default: 6)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Maximum beam search depth (default: 10)"
    )
    parser.add_argument(
        "--quant-configs", type=str, default="4,64;3,64;2,64",
        help="Semicolon-separated quantization configs as 'nbits,group_size' (default: '4,64;3,64;2,64')"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=20,
        help="Number of prompts to evaluate (default: 20)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--page-len", type=int, default=16,
        help="KV cache page length (default: 16)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=4096,
        help="Maximum number of KV cache pages (default: 4096)"
    )
    parser.add_argument(
        "--draft-device", type=str, default=None,
        help="Device for draft model (default: same as --device)"
    )
    parser.add_argument(
        "--share-kv", action="store_true",
        help="Share target model's prefill KV cache with draft (subspec-style)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Beam search temperature (default: 1.0, SubSpec uses 0.2)"
    )
    parser.add_argument(
        "--no-cuda-graph", action="store_true",
        help="Disable CUDA graph capture for beam search decode steps (enabled by default)"
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=1,
        help="Number of warm-up iterations per quant config before timed runs (default: 1)"
    )
    parser.add_argument(
        "--prompt-lengths", type=str, default=None,
        help="Comma-separated token lengths for prompts (e.g., '16,128,256,1024,4096,16384,65536'). "
             "Downloads a long text and slices to exact token lengths. Overrides --num-prompts."
    )
    parser.add_argument(
        "--prompt-source-url", type=str, default=GUTENBERG_DEFAULT_URL,
        help="URL to download source text for --prompt-lengths (default: War and Peace from Gutenberg)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for results (default: stdout)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    quant_configs = parse_quant_configs(args.quant_configs)

    if args.prompt_lengths:
        # Load tokenizer early to create length-specific prompts
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        token_lengths = [int(x.strip()) for x in args.prompt_lengths.split(",")]
        length_prompts = download_length_prompts(
            tokenizer, token_lengths, url=args.prompt_source_url
        )
        prompts = []
        for tok_len, prompt_str in length_prompts:
            logging.info(f"Prompt: {tok_len} tokens")
            prompts.append(prompt_str)
    else:
        prompts = DEFAULT_PROMPTS[:args.num_prompts]
        if len(prompts) < args.num_prompts:
            logging.warning(
                f"Only {len(prompts)} default prompts available (requested {args.num_prompts})"
            )

    logging.info(f"Model: {args.model}")
    logging.info(f"Beam width: {args.beam_width}, Max depth: {args.max_depth}")
    logging.info(f"Quant configs: {quant_configs}")
    logging.info(f"Prompts: {len(prompts)}")
    logging.info(f"Draft device: {args.draft_device or args.device}")
    logging.info(f"Share KV: {args.share_kv}")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"CUDA graph: {not args.no_cuda_graph}")
    logging.info(f"Warm-up iters: {args.warmup_iters}")

    experiment = BeamSearchExperiment(
        model_name=args.model,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        device=args.device,
        draft_device=args.draft_device,
        page_len=args.page_len,
        max_pages=args.max_pages,
        share_kv=args.share_kv,
        temperature=args.temperature,
        use_cuda_graph=not args.no_cuda_graph,
        warmup_iters=args.warmup_iters,
    )

    results = experiment.run_sweep(prompts, quant_configs)
    summary = results.format_summary()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(summary)

    if args.output:
        with open(args.output, "w") as f:
            f.write(summary)
        logging.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
