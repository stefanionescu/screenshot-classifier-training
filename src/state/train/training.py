"""Core training state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch
    from torch import nn
    from pathlib import Path
    from torch.utils.data import DataLoader
    from src.state.train.sampler.adaptive import AdaptiveSampler
    from src.state.train.types import SampleMeta, TrainDataset, TrainModel
    from src.state.contracts import DatasetSplit, ExportCheckpoint, ExportFormat

type BatchData = tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[SampleMeta, ...]]


@dataclass(frozen=True)
class TrainArgs:
    """Training CLI arguments."""

    dataset: str
    model: str
    screen_labels: list[str] | None
    image_size: int
    output_dir: str
    repo: str | None
    push: bool
    push_only: bool
    public: bool
    epochs: int
    micro_batch_size: int
    workers: int
    lr: float
    weight_decay: float
    grad_accum_steps: int
    batch_size: int
    seed: int
    eval_class_limit: int
    min_train_count: int
    screen_target_ratio: float
    safety_target_ratio: float
    screen_max_repeat: int
    safety_max_repeat: int
    safety_loss_weight: float
    safety_batch_probability: float
    export: ExportFormat
    resume: bool
    export_checkpoint: ExportCheckpoint


@dataclass(frozen=True)
class Sample:
    """One dataset image sample."""

    tar_path: Path
    image_name: str
    split: DatasetSplit
    screen_label: str
    safety_label: str
    width: int
    height: int


class SkippedImageRow(TypedDict):
    """Validated skipped-image journal record."""

    schema: int
    split: DatasetSplit
    tar_path: str
    image_member: str
    screen_label: str
    safety_label: str
    width: int
    height: int
    error: str


@dataclass(frozen=True)
class LabelState:
    """Trainable label names and id maps."""

    screen: list[str]
    safety: list[str]
    screen_to_id: dict[str, int]
    safety_to_id: dict[str, int]


@dataclass(frozen=True)
class RawSplits:
    """Raw dataset splits."""

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


@dataclass(frozen=True)
class SampleSplits:
    """Filtered dataset splits."""

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


@dataclass(frozen=True)
class ScreenCounts:
    """Screen label counts by split."""

    train: dict[str, int]
    val: dict[str, int]
    test: dict[str, int]


@dataclass(frozen=True)
class SafetyCounts:
    """Safety label counts by split."""

    train: dict[str, int]
    val: dict[str, int]
    test: dict[str, int]


@dataclass(frozen=True)
class DatasetBundle:
    """Dataset objects by split."""

    train: TrainDataset
    val: TrainDataset
    test: TrainDataset


@dataclass(frozen=True)
class SampleIds:
    """Sampler row ids by split and head."""

    train_screen: list[int]
    train_safety: list[int]
    val_screen: list[int]
    test_screen: list[int]
    val_safety: list[int]
    test_safety: list[int]


@dataclass(frozen=True)
class BatchIterators:
    """Batch iterators and adaptive samplers."""

    train_screen: DataLoader[BatchData | None]
    train_safety: DataLoader[BatchData | None]
    screen_sampler: AdaptiveSampler
    safety_sampler: AdaptiveSampler
    ids: SampleIds
    val: DataLoader[BatchData | None]
    test: DataLoader[BatchData | None]
    val_screen: DataLoader[BatchData | None]
    test_screen: DataLoader[BatchData | None]
    val_safety: DataLoader[BatchData | None]
    test_safety: DataLoader[BatchData | None]


@dataclass(frozen=True)
class MetricIterators:
    """Balanced validation and test iterators for both output heads."""

    val_screen: DataLoader[BatchData | None]
    test_screen: DataLoader[BatchData | None]
    val_safety: DataLoader[BatchData | None]
    test_safety: DataLoader[BatchData | None]


@dataclass
class ModelState:
    """Model, optimizer, scheduler, and checkpoint position."""

    model: TrainModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    scaler: torch.GradScaler
    screen_criterion: nn.Module
    safety_criterion: nn.Module
    start_epoch: int
    best_score: float
    best_epoch: int
