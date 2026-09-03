"""Training data state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch
    from pathlib import Path
    from src.state.dashboard import TrainDashboardProtocol
    from src.state.types import NormalizeStats, TrainDataset
    from src.state.training import LabelState, SafetyCounts, Sample, SampleSplits, ScreenCounts, TrainArgs


@dataclass(frozen=True)
class PreprocessSpec:
    """Image normalization parameters."""

    mean: NormalizeStats
    std: NormalizeStats


@dataclass(frozen=True)
class DatasetSpec:
    """Training dataset settings."""

    samples: list[Sample]
    labels: LabelState
    image_size: int
    preprocess: PreprocessSpec
    augment: bool
    skipped_path: Path | None = None


@dataclass(frozen=True)
class TrainConfig:
    """Training config fields."""

    args: TrainArgs
    model_id: str
    dataset_path: Path
    labels: LabelState
    counts: ScreenCounts
    safety_counts: SafetyCounts
    fold_map: dict[str, str]
    preprocess: PreprocessSpec
    accumulation_steps: int


@dataclass(frozen=True)
class DatasetBuild:
    """Inputs for building datasets."""

    samples: SampleSplits
    labels: LabelState
    args: TrainArgs
    preprocess: PreprocessSpec
    skipped_path: Path
    dashboard: TrainDashboardProtocol


@dataclass(frozen=True)
class IteratorConfig:
    """Adaptive train iterator settings."""

    dataset: TrainDataset
    labels: list[int]
    target_ratio: float
    max_repeat: int
    seed: int
    batch_size: int
    workers: int
    run_device: torch.device
