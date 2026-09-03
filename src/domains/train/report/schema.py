"""Validate persisted training metrics at the JSON boundary."""

from __future__ import annotations

import math
from typing import cast
from src.errors import TrainingError
from src.state.train.metrics import (
    HeadMetrics,
    ClassMetrics,
    LatencyStats,
    TimingMetrics,
    EvaluationMetrics,
    ValidationMetrics,
)

HEAD_FIELDS = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "top2_accuracy",
    "total",
    "per_class",
    "confusion_matrix",
}
CLASS_FIELDS = {"precision", "recall", "f1", "support"}
TIMING_FIELDS = {
    "format",
    "provider",
    "device",
    "hardware",
    "image_size",
    "batch_size",
    "image_count",
    "images_per_second",
    "read_preprocess",
    "model_run",
    "total",
}


def record(value: object, name: str) -> dict[str, object]:
    """Return a string-keyed JSON object or raise a report error."""
    if not isinstance(value, dict):
        msg = f"training report {name} must be a JSON object."
        raise TrainingError(msg)
    values = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in values):
        msg = f"training report {name} must be a JSON object."
        raise TrainingError(msg)
    return cast("dict[str, object]", values)


def finite_number(value: object, name: str) -> float:
    """Return a finite non-Boolean number."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        msg = f"training report {name} must be a finite number."
        raise TrainingError(msg)
    return float(value)


def nonnegative_integer(value: object, name: str) -> int:
    """Return a non-negative non-Boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"training report {name} must be a non-negative integer."
        raise TrainingError(msg)
    return value


def string_value(value: object, name: str) -> str:
    """Return a non-empty string."""
    if not isinstance(value, str) or not value:
        msg = f"training report {name} must be a non-empty string."
        raise TrainingError(msg)
    return value


def parse_class_metrics(value: object, name: str) -> ClassMetrics:
    """Validate one class metric record."""
    values = record(value, name)
    if set(values) != CLASS_FIELDS:
        msg = f"training report {name} fields do not match the metric schema."
        raise TrainingError(msg)
    return ClassMetrics(
        precision=finite_number(values["precision"], f"{name}.precision"),
        recall=finite_number(values["recall"], f"{name}.recall"),
        f1=finite_number(values["f1"], f"{name}.f1"),
        support=nonnegative_integer(values["support"], f"{name}.support"),
    )


def valid_confusion_row(value: object) -> bool:
    """Return whether a confusion-matrix row contains nonnegative integers."""
    if not isinstance(value, list):
        return False
    items = cast("list[object]", value)
    return all(not isinstance(item, bool) and isinstance(item, int) and item >= 0 for item in items)


def parse_head_metrics(value: object, name: str) -> HeadMetrics:
    """Validate one output-head metric record."""
    values = record(value, name)
    if set(values) != HEAD_FIELDS:
        msg = f"training report {name} fields do not match the head schema."
        raise TrainingError(msg)
    classes = record(values["per_class"], f"{name}.per_class")
    confusion = record(values["confusion_matrix"], f"{name}.confusion_matrix")
    labels = confusion.get("labels")
    matrix = confusion.get("matrix")
    if set(confusion) != {"labels", "matrix"} or not isinstance(labels, list) or not isinstance(matrix, list):
        msg = f"training report {name}.confusion_matrix is invalid."
        raise TrainingError(msg)
    label_values = cast("list[object]", labels)
    matrix_values = cast("list[object]", matrix)
    if any(not isinstance(label, str) or not label for label in label_values):
        msg = f"training report {name}.confusion_matrix labels are invalid."
        raise TrainingError(msg)
    if any(not valid_confusion_row(row) for row in matrix_values):
        msg = f"training report {name}.confusion_matrix values are invalid."
        raise TrainingError(msg)
    return HeadMetrics(
        accuracy=finite_number(values["accuracy"], f"{name}.accuracy"),
        balanced_accuracy=finite_number(values["balanced_accuracy"], f"{name}.balanced_accuracy"),
        macro_f1=finite_number(values["macro_f1"], f"{name}.macro_f1"),
        top2_accuracy=finite_number(values["top2_accuracy"], f"{name}.top2_accuracy"),
        total=nonnegative_integer(values["total"], f"{name}.total"),
        per_class={
            label: parse_class_metrics(metrics, f"{name}.per_class.{label}") for label, metrics in classes.items()
        },
        confusion_matrix={"labels": cast("list[str]", labels), "matrix": cast("list[list[int]]", matrix)},
    )


def parse_latency(value: object, name: str) -> LatencyStats:
    """Validate one latency distribution."""
    values = record(value, name)
    if set(values) != {"mean_ms", "median_ms", "p95_ms"}:
        msg = f"training report {name} fields do not match the latency schema."
        raise TrainingError(msg)
    return LatencyStats(
        mean_ms=finite_number(values["mean_ms"], f"{name}.mean_ms"),
        median_ms=finite_number(values["median_ms"], f"{name}.median_ms"),
        p95_ms=finite_number(values["p95_ms"], f"{name}.p95_ms"),
    )


def parse_timing(value: object, name: str) -> TimingMetrics:
    """Validate measured timing context and distributions."""
    values = record(value, name)
    if set(values) != TIMING_FIELDS:
        msg = f"training report {name} fields do not match the timing schema."
        raise TrainingError(msg)
    return TimingMetrics(
        format=string_value(values["format"], f"{name}.format"),
        provider=string_value(values["provider"], f"{name}.provider"),
        device=string_value(values["device"], f"{name}.device"),
        hardware=string_value(values["hardware"], f"{name}.hardware"),
        image_size=nonnegative_integer(values["image_size"], f"{name}.image_size"),
        batch_size=nonnegative_integer(values["batch_size"], f"{name}.batch_size"),
        image_count=nonnegative_integer(values["image_count"], f"{name}.image_count"),
        images_per_second=finite_number(values["images_per_second"], f"{name}.images_per_second"),
        read_preprocess=parse_latency(values["read_preprocess"], f"{name}.read_preprocess"),
        model_run=parse_latency(values["model_run"], f"{name}.model_run"),
        total=parse_latency(values["total"], f"{name}.total"),
    )


def parse_evaluation(value: object, name: str) -> EvaluationMetrics:
    """Validate one screen-and-safety evaluation."""
    values = record(value, name)
    if set(values) not in ({"screen", "safety"}, {"screen", "safety", "timing"}):
        msg = f"training report {name} fields do not match the evaluation schema."
        raise TrainingError(msg)
    result = EvaluationMetrics(
        screen=parse_head_metrics(values["screen"], f"{name}.screen"),
        safety=parse_head_metrics(values["safety"], f"{name}.safety"),
    )
    if "timing" in values:
        result["timing"] = parse_timing(values["timing"], f"{name}.timing")
    return result


def parse_validation_metrics(value: object) -> ValidationMetrics:
    """Read the three reusable validation evaluations from a report object."""
    report = record(value, "root")
    return ValidationMetrics(
        full=parse_evaluation(report.get("val_full"), "val_full"),
        screen_balanced=parse_evaluation(report.get("val_screen_balanced"), "val_screen_balanced"),
        safety_balanced=parse_evaluation(report.get("val_safety_balanced"), "val_safety_balanced"),
    )
