"""
Sparse attention strategies for beam search decode.

Factory function to create strategies from config.
"""

from __future__ import annotations

from .base import SparseAttentionConfig, SparseAttentionStrategy
from .noop import NoopStrategy
from .pillar import PillarStrategy


def create_strategy(config: SparseAttentionConfig) -> SparseAttentionStrategy:
    """Create a sparse attention strategy from config.

    Args:
        config: SparseAttentionConfig specifying method and parameters.

    Returns:
        Concrete SparseAttentionStrategy instance.
    """
    if not config.enabled or config.method == "none":
        return NoopStrategy(config)
    elif config.method == "pillar":
        return PillarStrategy(config)
    else:
        raise ValueError(f"Unknown sparse attention method: {config.method!r}")


__all__ = [
    "SparseAttentionConfig",
    "SparseAttentionStrategy",
    "NoopStrategy",
    "PillarStrategy",
    "create_strategy",
]
