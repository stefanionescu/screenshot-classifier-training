"""Training checkpoint state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch
    from pathlib import Path
    from src.state.train.types import TrainModel
    from src.state.train.training import TrainArgs
    from src.state.train.sampler.adaptive import AdaptiveSampler


@dataclass(frozen=True)
class Checkpoint:
    """Checkpoint to save."""

    path: Path
    model: TrainModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    scaler: torch.GradScaler
    epoch: int
    best_score: float
    best_epoch: int
    screen_labels: list[str]
    safety_labels: list[str]
    screen_sampler: AdaptiveSampler
    safety_sampler: AdaptiveSampler
    args: TrainArgs
    run_dir: Path


@dataclass(frozen=True)
class SavedCheckpoint:
    """Checkpoint to load."""

    path: Path
    model: TrainModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    scaler: torch.GradScaler
    run_device: torch.device
    screen_labels: list[str]
    safety_labels: list[str]
    screen_sampler: AdaptiveSampler
    safety_sampler: AdaptiveSampler
    model_id: str
    image_size: int
    train_config_sha256: str


@dataclass(frozen=True)
class ModelCheckpoint:
    """Model checkpoint to load."""

    path: Path
    model: TrainModel
    run_device: torch.device
    screen_labels: list[str]
    safety_labels: list[str]
