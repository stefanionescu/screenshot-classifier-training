"""Typed training evaluation and report records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from src.state.sampler.adaptive import SamplerCoverage


class ClassMetrics(TypedDict):
    """Precision, recall, F1, and support for one label."""

    precision: float
    recall: float
    f1: float
    support: int


class ConfusionMatrix(TypedDict):
    """Label order and integer confusion matrix."""

    labels: list[str]
    matrix: list[list[int]]


class HeadMetrics(TypedDict):
    """Metrics for one classification head."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    top2_accuracy: float
    total: int
    per_class: dict[str, ClassMetrics]
    confusion_matrix: ConfusionMatrix


class LatencyStats(TypedDict):
    """Latency distribution in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float


class TimingMetrics(TypedDict):
    """Evaluation throughput and latency context."""

    format: NotRequired[str]
    provider: str
    device: str
    hardware: NotRequired[str]
    image_size: int
    batch_size: int
    image_count: int
    images_per_second: float
    read_preprocess: LatencyStats
    model_run: LatencyStats
    total: LatencyStats


class TimingContext(TypedDict):
    """Static context attached to measured model timing."""

    format: str
    provider: str
    device: str
    hardware: str
    image_size: int
    batch_size: int


class EvaluationMetrics(TypedDict):
    """Screen, safety, and optional timing evaluation metrics."""

    screen: HeadMetrics
    safety: HeadMetrics
    timing: NotRequired[TimingMetrics]


class ModelEvaluationMetrics(TypedDict):
    """Test metrics for one exported model format."""

    test_full: EvaluationMetrics
    test_screen_balanced: EvaluationMetrics
    test_safety_balanced: EvaluationMetrics


class TrainingProfile(TypedDict):
    """Dataset counts and warnings captured before training."""

    split_counts: dict[str, int]
    screen_counts: dict[str, dict[str, int]]
    safety_counts: dict[str, dict[str, int]]
    screen_by_safety: dict[str, dict[str, dict[str, int]]]
    size_buckets: dict[str, dict[str, dict[str, int]]]
    warnings: list[str]


class SamplerReport(TypedDict):
    """Persisted adaptive-sampler configuration and coverage."""

    mode: Literal["adaptive_label_aspect_bucketed"]
    screen_train_batches: int
    safety_train_batches: int
    screen_labels: list[str]
    safety_labels: list[str]
    screen_coverage: SamplerCoverage
    safety_coverage: SamplerCoverage


class SkippedImagesSummary(TypedDict):
    """Skipped-image journal summary."""

    file: str
    total: int
    counts: dict[str, int]


@dataclass(frozen=True)
class ValidationMetrics:
    """Validation evaluations used by reporting and checkpoint resume."""

    full: EvaluationMetrics
    screen_balanced: EvaluationMetrics
    safety_balanced: EvaluationMetrics


@dataclass(frozen=True)
class EvaluationSummary:
    """Complete validation and exported-model evaluations."""

    validation: ValidationMetrics
    test: ModelEvaluationMetrics
    onnx_models: dict[str, ModelEvaluationMetrics]


class TrainingReport(TypedDict):
    """Persisted final training report."""

    model: str
    image_size: int
    screen_labels: list[str]
    safety_labels: list[str]
    best_checkpoint: str
    profile: TrainingProfile
    sampling: SamplerReport
    skipped_images: SkippedImagesSummary
    val_full: EvaluationMetrics
    val_screen_balanced: EvaluationMetrics
    val_safety_balanced: EvaluationMetrics
    test_full: EvaluationMetrics
    test_screen_balanced: EvaluationMetrics
    test_safety_balanced: EvaluationMetrics
    onnx_models: dict[str, ModelEvaluationMetrics]
