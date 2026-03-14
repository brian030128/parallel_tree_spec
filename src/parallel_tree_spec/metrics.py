"""
Metrics collection and reporting for speculative decoding experiments.

Tracks per-run and aggregated metrics: acceptance counts, per-depth acceptance,
draft step times, and verification times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class SingleRunMetrics:
    """Metrics from one draft+verify cycle."""
    accept_len: int                          # accepted tokens (excludes bonus)
    total_len: int                           # total verification steps
    draft_step_times: List[float]            # per-step decode times (seconds)
    verify_time: float                       # target forward + verify time (seconds)
    tree_size: int                           # total nodes in draft tree
    tree_depth: int                          # max depth of draft tree
    target_decode_time: float = 0.0              # single-token target decode (seconds)
    per_depth_accepted: Dict[int, bool] = field(default_factory=dict)
    # depth -> was token at this depth accepted?
    prompt_length: int = 0                       # token length of prompt (0 = unknown)
    sparse_method: str = "none"                  # sparse attention method used


@dataclass
class QuantConfigResult:
    """Aggregated results for one quantization config across multiple prompts."""
    nbits: int
    group_size: int
    runs: List[SingleRunMetrics] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    @property
    def mean_accept_len(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.accept_len for r in self.runs) / len(self.runs)

    @property
    def mean_draft_time(self) -> float:
        """Mean total draft time per run (excludes prefill/setup at index 0)."""
        if not self.runs:
            return 0.0
        return sum(sum(r.draft_step_times[1:]) for r in self.runs) / len(self.runs)

    @property
    def mean_verify_time(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.verify_time for r in self.runs) / len(self.runs)

    @property
    def mean_target_decode_time(self) -> float:
        """Mean single-token target decode time."""
        if not self.runs:
            return 0.0
        return sum(r.target_decode_time for r in self.runs) / len(self.runs)

    @property
    def mean_step_time(self) -> float:
        """Mean per-step draft decode time across all runs and steps.

        Excludes index 0 (prefill/setup overhead, not a real decode step).
        """
        all_times = [t for r in self.runs for t in r.draft_step_times[1:]]
        if not all_times:
            return 0.0
        return sum(all_times) / len(all_times)

    def per_depth_acceptance_rate(self, max_depth: int) -> Dict[int, float]:
        """Acceptance rate at each depth across all runs.

        For each depth d (1..max_depth), compute the fraction of runs where
        the accepted path reached at least depth d.
        """
        rates = {}
        for d in range(1, max_depth + 1):
            count = sum(1 for r in self.runs if r.accept_len >= d)
            rates[d] = count / len(self.runs) if self.runs else 0.0
        return rates

    def step_times_by_index(self) -> Dict[int, List[float]]:
        """Group step times by step index across runs."""
        result: Dict[int, List[float]] = {}
        for r in self.runs:
            for i, t in enumerate(r.draft_step_times):
                result.setdefault(i, []).append(t)
        return result

    def runs_by_prompt_length(self) -> Dict[int, List[SingleRunMetrics]]:
        """Group runs by prompt_length. Only includes runs with prompt_length > 0."""
        groups: Dict[int, List[SingleRunMetrics]] = {}
        for r in self.runs:
            if r.prompt_length > 0:
                groups.setdefault(r.prompt_length, []).append(r)
        return groups

    @staticmethod
    def _summarize_runs(runs: List[SingleRunMetrics]) -> dict:
        """Compute mean stats from a list of runs."""
        n = len(runs)
        if n == 0:
            return {}
        return {
            "n": n,
            "accept": sum(r.accept_len for r in runs) / n,
            "draft_ms": sum(sum(r.draft_step_times[1:]) for r in runs) / n * 1000,
            "step_ms": sum(t for r in runs for t in r.draft_step_times[1:])
                       / max(sum(len(r.draft_step_times[1:]) for r in runs), 1) * 1000,
            "verify_ms": sum(r.verify_time for r in runs) / n * 1000,
            "target_ms": sum(r.target_decode_time for r in runs) / n * 1000,
        }


@dataclass
class SweepResults:
    """Results from a full sweep across quantization configs."""
    model_name: str
    beam_width: int
    max_depth: int
    configs: List[QuantConfigResult] = field(default_factory=list)

    def format_summary(self) -> str:
        """Format a human-readable summary table."""
        lines = []
        lines.append(f"Model: {self.model_name}")
        lines.append(f"Beam width: {self.beam_width}, Max depth: {self.max_depth}")
        lines.append("")

        # Header
        header = f"{'Config':>12s} | {'Runs':>5s} | {'Accept':>7s} | {'Draft(ms)':>10s} | {'Step(ms)':>9s} | {'Verify(ms)':>11s} | {'Target(ms)':>11s}"
        lines.append(header)
        lines.append("-" * len(header))

        for cfg in self.configs:
            label = f"{cfg.nbits}b/g{cfg.group_size}"
            lines.append(
                f"{label:>12s} | {cfg.num_runs:>5d} | "
                f"{cfg.mean_accept_len:>7.2f} | "
                f"{cfg.mean_draft_time * 1000:>10.2f} | "
                f"{cfg.mean_step_time * 1000:>9.2f} | "
                f"{cfg.mean_verify_time * 1000:>11.2f} | "
                f"{cfg.mean_target_decode_time * 1000:>11.2f}"
            )

        lines.append("")

        # Per-depth acceptance rates
        for cfg in self.configs:
            label = f"{cfg.nbits}b/g{cfg.group_size}"
            rates = cfg.per_depth_acceptance_rate(self.max_depth)
            lines.append(f"Per-depth acceptance rate ({label}):")
            depth_strs = [f"  d={d}: {rate:.1%}" for d, rate in rates.items()]
            lines.append("  " + "  ".join(depth_strs[:5]))
            if len(depth_strs) > 5:
                lines.append("  " + "  ".join(depth_strs[5:]))
            lines.append("")

        # Per-step draft timing
        for cfg in self.configs:
            label = f"{cfg.nbits}b/g{cfg.group_size}"
            step_times = cfg.step_times_by_index()
            if step_times:
                lines.append(f"Per-step draft decode time ({label}):")
                for idx in sorted(step_times.keys()):
                    if idx == 0:
                        continue  # skip prefill/setup overhead
                    times = step_times[idx]
                    mean_t = sum(times) / len(times)
                    lines.append(f"  step {idx}: {mean_t * 1000:.2f} ms")
                lines.append("")

        # Per-prompt-length breakdown (only when runs have prompt_length > 0)
        for cfg in self.configs:
            groups = cfg.runs_by_prompt_length()
            if not groups:
                continue
            label = f"{cfg.nbits}b/g{cfg.group_size}"
            lines.append(f"Per prompt-length breakdown ({label}):")
            pl_header = (
                f"  {'Prompt Len':>10s} | {'Runs':>5s} | {'Accept':>6s} | "
                f"{'Draft(ms)':>9s} | {'Step(ms)':>8s} | {'Verify(ms)':>10s} | {'Target(ms)':>10s}"
            )
            lines.append(pl_header)
            lines.append("  " + "-" * (len(pl_header) - 2))
            for plen in sorted(groups.keys()):
                s = QuantConfigResult._summarize_runs(groups[plen])
                lines.append(
                    f"  {plen:>10d} | {s['n']:>5d} | {s['accept']:>6.2f} | "
                    f"{s['draft_ms']:>9.2f} | {s['step_ms']:>8.2f} | "
                    f"{s['verify_ms']:>10.2f} | {s['target_ms']:>10.2f}"
                )
            lines.append("")

        return "\n".join(lines)
