"""Adaptive sampler checkpoint state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplerSnapshot:
    """Label orders, cursors, cycles, and epoch restored from a checkpoint."""

    epoch: int
    orders: dict[int, list[int]]
    cursors: dict[int, int]
    cycles: dict[int, int]
