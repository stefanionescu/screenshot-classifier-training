"""Validated random-number generator checkpoint state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch
    import numpy as np


@dataclass(frozen=True)
class NumpyStateSnapshot:
    """Validated NumPy random-number state."""

    name: str
    keys: np.ndarray[tuple[int], np.dtype[np.uint32]]
    position: int
    has_gauss: int
    cached_gaussian: float


@dataclass(frozen=True)
class RandomStateSnapshot:
    """Restorable Python, NumPy, Torch CPU, and Torch CUDA states."""

    python: tuple[object, ...]
    numpy_name: str
    numpy_keys: np.ndarray[tuple[int], np.dtype[np.uint32]]
    numpy_position: int
    numpy_has_gauss: int
    numpy_cached_gaussian: float
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]
