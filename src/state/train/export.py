"""Training export and orchestration state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch
    from pathlib import Path
    from src.state.train.types import TrainModel
    from src.state.train.recipe import TrainingRecipe
    from src.state.train.dashboard import TrainDashboardProtocol
    from src.state.train.metrics import EvaluationSummary, TrainingProfile
    from src.state.train.training import (
        SampleIds,
        TrainArgs,
        LabelState,
        DatasetBundle,
        BatchIterators,
    )


class ExportPreprocess(TypedDict):
    """Exported image preprocessing contract."""

    resize_longest_side_px: int
    preserve_aspect_ratio: bool
    crop: bool
    stretch: bool
    horizontal_flip: bool
    mean: list[float]
    std: list[float]


class ExportModelConfig(TypedDict):
    """Exported classifier architecture contract."""

    name: str
    architecture: str
    model: str
    outputs: list[str]
    screen_labels: list[str]
    safety_labels: list[str]
    resize_longest_side_px: int


class ExportLabelHead(TypedDict):
    """Labels and stable IDs for one exported output head."""

    labels: list[str]
    label_to_id: dict[str, int]


class ExportLabels(TypedDict):
    """Exported labels for both classifier output heads."""

    screen: ExportLabelHead
    safety: ExportLabelHead


@dataclass(frozen=True)
class ModelExport:
    """Model export files."""

    model: TrainModel
    export_dir: Path
    model_id: str
    labels: LabelState
    image_size: int
    args: TrainArgs


@dataclass(frozen=True)
class ModelCard:
    """Model card content."""

    export_dir: Path
    model_id: str
    repo_id: str
    report: EvaluationSummary | None = None


@dataclass(frozen=True)
class TrainJob:
    """Training job."""

    args: TrainArgs
    token: str | None
    model_id: str
    run_dir: Path
    export_dir: Path
    repo_id: str
    saved_config: TrainingRecipe | None
    dashboard: TrainDashboardProtocol


@dataclass(frozen=True)
class PreparedTraining:
    """Prepared labels, profile, ids, and datasets."""

    labels: LabelState
    profile: TrainingProfile
    ids: SampleIds
    datasets: DatasetBundle


@dataclass(frozen=True)
class ModelRun:
    """Training model run."""

    iterators: BatchIterators
    model: TrainModel
    run_device: torch.device
    eval_only: bool


@dataclass(frozen=True)
class ValMetrics:
    """Validation metrics fields."""

    eval_only: bool
    run_dir: Path
    model: TrainModel
    labels: LabelState
    iterators: BatchIterators
    run_device: torch.device
    dashboard: TrainDashboardProtocol


@dataclass(frozen=True)
class OnnxMetrics:
    """ONNX metrics fields."""

    export_dir: Path
    run_dir: Path
    labels: LabelState
    iterators: BatchIterators
    args: TrainArgs
    dashboard: TrainDashboardProtocol


@dataclass(frozen=True)
class OnnxModelMetrics:
    """ONNX model metrics fields."""

    model_path: Path
    model_format: str
    model_key: str
    metrics: OnnxMetrics


@dataclass(frozen=True)
class ReportInputs:
    """Training report fields."""

    args: TrainArgs
    model_id: str
    run_dir: Path
    profile: TrainingProfile
    labels: LabelState
    iterators: BatchIterators
    evaluation_summary: EvaluationSummary
