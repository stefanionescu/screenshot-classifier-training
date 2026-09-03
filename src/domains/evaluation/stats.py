"""Training metric calculations."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, cast
from src.config.metrics import CHECKPOINT_ACCURACY_WEIGHT
from src.config.model import SCREEN_FALLBACK_LABEL, SCREEN_OTHER_LABEL
from src.state.metrics import (
    HeadMetrics,
    ClassMetrics,
    LatencyStats,
    ConfusionMatrix,
    EvaluationMetrics,
)

if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray


def latency_stats(seconds: list[float]) -> LatencyStats:
    """Return latency summary values."""
    if not seconds:
        return LatencyStats(mean_ms=0.0, median_ms=0.0, p95_ms=0.0)
    return LatencyStats(
        mean_ms=1000 * sum(seconds) / len(seconds),
        median_ms=1000 * float(np.median(seconds)),
        p95_ms=1000 * float(np.quantile(seconds, 0.95, method="inverted_cdf")),
    )


def metrics_from_confusion(
    confusion: torch.Tensor,
    labels: list[str],
    top2_hits: int | None,
    total: int,
) -> HeadMetrics:
    """Return metrics from a confusion matrix."""
    per_class: dict[str, ClassMetrics] = {}
    correct = int(confusion.diag().sum())
    matrix = cast("NDArray[np.int64]", confusion.cpu().numpy())
    precisions, recalls, f1_scores, supports = per_class_scores(matrix, len(labels))
    for index, label in enumerate(labels):
        per_class[label] = ClassMetrics(
            precision=float(precisions[index]),
            recall=float(recalls[index]),
            f1=float(f1_scores[index]),
            support=int(supports[index]),
        )
    confusion_matrix = cast("list[list[int]]", matrix.astype(int).tolist())
    return HeadMetrics(
        accuracy=safe_divide(correct, total),
        balanced_accuracy=float(np.mean(recalls)) if len(recalls) else 0.0,
        macro_f1=float(np.mean(f1_scores)) if len(f1_scores) else 0.0,
        top2_accuracy=safe_divide(top2_hits, total) if top2_hits is not None else 0.0,
        total=total,
        per_class=per_class,
        confusion_matrix=ConfusionMatrix(labels=labels, matrix=confusion_matrix),
    )


def per_class_scores(
    matrix: NDArray[np.int64],
    label_count: int,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Return precision, recall, F1, and support for each label."""
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    supports: list[int] = []
    for index in range(label_count):
        true_positive = int(matrix[index, index])
        predicted_total = int(matrix[:, index].sum())
        support = int(matrix[index, :].sum())
        precision = safe_divide(true_positive, predicted_total)
        recall = safe_divide(true_positive, support)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(safe_divide(2.0 * precision * recall, precision + recall))
        supports.append(support)
    return precisions, recalls, f1_scores, supports


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide and return zero for empty denominators."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def checkpoint_score(screen_metrics: EvaluationMetrics, safety_metrics: EvaluationMetrics) -> float:
    """Return the validation checkpoint score."""
    screen = screen_metrics["screen"]
    safety = safety_metrics["safety"]
    screen_classes = screen["per_class"]
    fallback_recall = screen_classes.get(SCREEN_FALLBACK_LABEL, ClassMetrics(precision=0, recall=0, f1=0, support=0))[
        "recall"
    ]
    other_recall = screen_classes.get(SCREEN_OTHER_LABEL, ClassMetrics(precision=0, recall=0, f1=0, support=0))[
        "recall"
    ]
    return float(
        CHECKPOINT_ACCURACY_WEIGHT * screen["accuracy"]
        + screen["macro_f1"]
        + fallback_recall
        + other_recall
        + CHECKPOINT_ACCURACY_WEIGHT * safety["accuracy"]
        + safety["macro_f1"]
        + safety["balanced_accuracy"],
    )
