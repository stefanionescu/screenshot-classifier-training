"""Training loop state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch
    from pathlib import Path
    from src.state.train.types import TrainModel
    from src.state.train.dashboard import TrainDashboardProtocol
    from src.state.train.training import BatchIterators, LabelState, ModelState, TrainArgs


@dataclass(frozen=True)
class TrainBatch:
    """One batch on the target device."""

    images: torch.Tensor
    screen_targets: torch.Tensor
    safety_targets: torch.Tensor
    image_count: int


@dataclass(frozen=True)
class BatchLosses:
    """Loss tensors from one batch."""

    loss: torch.Tensor
    screen: torch.Tensor
    safety: torch.Tensor


@dataclass(frozen=True)
class TrainRun:
    """Shared state for epoch training helpers."""

    args: TrainArgs
    labels: LabelState
    iterators: BatchIterators
    state: ModelState
    run_device: torch.device
    run_dir: Path
    dashboard: TrainDashboardProtocol


@dataclass(frozen=True)
class EvalRun:
    """Shared state for evaluation metric helpers."""

    model: TrainModel
    labels: LabelState
    iterators: BatchIterators
    run_device: torch.device
    dashboard: TrainDashboardProtocol
    run_dir: Path | None = None
