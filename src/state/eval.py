"""Training evaluation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    import torch
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from src.state.metrics import TimingContext
    from src.state.contracts import DatasetSplit
    from collections.abc import Mapping, Sequence
    from src.state.types import SampleMeta, TrainModel
    from src.state.training import BatchData, LabelState
    from src.state.dashboard import TrainDashboardProtocol


class OnnxSession(Protocol):
    """ONNX inference session shape used by training evaluation."""

    def get_inputs(self) -> Sequence[SimpleNamespace]:
        """Return input metadata."""
        raise NotImplementedError

    def get_outputs(self) -> Sequence[SimpleNamespace]:
        """Return output metadata."""
        raise NotImplementedError

    def run(self, output_names: Sequence[str] | None, _input_feed: Mapping[str, object]) -> Sequence[np.ndarray]:
        """Run inference and return output arrays."""
        raise NotImplementedError(output_names, _input_feed)


@dataclass(frozen=True)
class EvalArtifacts:
    """Represent eval artifacts."""

    run_dir: Path
    stage: str
    split: DatasetSplit
    write_predictions: bool = True


@dataclass(frozen=True)
class PredBatch:
    """Prediction batch data."""

    metas: tuple[SampleMeta, ...]
    screen_labels: list[str]
    safety_labels: list[str]
    screen_logits: torch.Tensor | np.ndarray
    safety_logits: torch.Tensor | np.ndarray
    screen_targets: torch.Tensor | np.ndarray
    safety_targets: torch.Tensor | np.ndarray


class PredictionRow(TypedDict):
    """Persisted evaluation prediction record."""

    stage: str
    split: DatasetSplit
    tar_path: str
    image_member: str
    width: int
    height: int
    true_screen: str
    pred_screen: str
    screen_correct: bool
    screen_top2: list[str]
    screen_top2_hit: bool
    screen_confidence: float
    true_safety: str | None
    pred_safety: str
    safety_correct: bool | None
    safety_confidence: float
    screen_scores: dict[str, float]
    safety_scores: dict[str, float]


class ConfidenceBucket(TypedDict):
    """Calibration statistics for one confidence interval."""

    min: float
    max: float
    count: int
    accuracy: float
    wrong_count: int
    wrong_rate: float


class CalibrationThresholds(TypedDict):
    """Named diagnostic confidence thresholds."""

    confident_80: float
    confident_90: float
    unconfident: float


class CalibrationReport(TypedDict):
    """Screen and safety calibration buckets."""

    screen: list[ConfidenceBucket]
    safety: list[ConfidenceBucket]
    thresholds: CalibrationThresholds


@dataclass(frozen=True)
class TimedBatch:
    """One iterator result with measured iterator latency."""

    index: int
    batch: BatchData | None
    read_seconds: float


@dataclass(frozen=True)
class TorchEval:
    """Torch evaluation run."""

    model: TrainModel
    iterator: DataLoader[BatchData | None]
    run_device: torch.device
    labels: LabelState
    description: str
    dashboard: TrainDashboardProtocol | None = None
    timing: TimingContext | None = None
    artifacts: EvalArtifacts | None = None


@dataclass(frozen=True)
class OnnxEval:
    """ONNX evaluation run."""

    model_path: Path
    iterator: DataLoader[BatchData | None]
    screen_labels: list[str]
    safety_labels: list[str]
    image_size: int
    batch_size: int
    model_format: str
    dashboard: TrainDashboardProtocol | None = None
    stage: str | None = None
    artifacts: EvalArtifacts | None = None


@dataclass
class MetricCounts:
    """Metric counters."""

    screen_confusion: torch.Tensor
    safety_confusion: torch.Tensor
    screen_top2_hits: int = 0
    safety_top2_hits: int = 0
    screen_total: int = 0
    safety_total: int = 0


@dataclass
class LatencySamples:
    """Per-batch, per-image-normalized latency observations."""

    read: list[float] = field(default_factory=list)
    model: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    image_count: int = 0
    elapsed: float = 0.0


@dataclass(frozen=True)
class OnnxLoop:
    """ONNX loop state."""

    run: OnnxEval
    session: OnnxSession
