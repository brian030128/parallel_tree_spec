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
import sys
from typing import List, Tuple

from parallel_tree_spec.experiment import BeamSearchExperiment, DEFAULT_PROMPTS


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
    prompts = DEFAULT_PROMPTS[:args.num_prompts]

    if len(prompts) < args.num_prompts:
        logging.warning(
            f"Only {len(prompts)} default prompts available (requested {args.num_prompts})"
        )

    logging.info(f"Model: {args.model}")
    logging.info(f"Beam width: {args.beam_width}, Max depth: {args.max_depth}")
    logging.info(f"Quant configs: {quant_configs}")
    logging.info(f"Prompts: {len(prompts)}")
    logging.info(f"Share KV: {args.share_kv}")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"CUDA graph: {not args.no_cuda_graph}")
    logging.info(f"Warm-up iters: {args.warmup_iters}")

    experiment = BeamSearchExperiment(
        model_name=args.model,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        device=args.device,
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
