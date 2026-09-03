"""Training dashboard formatting helpers."""

from __future__ import annotations

from rich.table import Table
from typing import TYPE_CHECKING
from src.shared.runtime.format import format_elapsed_time
from src.config.train.metrics import (
    DASHBOARD_PROGRESS_WIDTH,
    DASHBOARD_SPARKLINE_WIDTH,
    DASHBOARD_SPARKLINE_BLOCKS,
    DASHBOARD_SMALL_FLOAT_THRESHOLD,
)

if TYPE_CHECKING:
    from src.state.train.metrics import (
        HeadMetrics,
        ClassMetrics,
        LatencyStats,
        TimingMetrics,
        EvaluationMetrics,
        EvaluationSummary,
        ModelEvaluationMetrics,
    )


def format_metric_value(number: object) -> str:
    """Format a metric value."""
    if isinstance(number, float):
        if abs(number) < DASHBOARD_SMALL_FLOAT_THRESHOLD and number != 0.0:
            return f"{number:.2e}"
        return f"{number:.4f}"
    if isinstance(number, int):
        return str(number)
    return "-"


def collect_class_recalls(classes: dict[str, ClassMetrics]) -> dict[str, float]:
    """Return recall values by class label."""
    return {label: metrics["recall"] for label, metrics in classes.items()}


def build_evaluation_rows(evaluation_summary: EvaluationSummary) -> tuple[tuple[str, ...], ...]:
    """Return final metric table rows."""
    rows: list[tuple[str, ...]] = []
    for model_name, metrics in evaluation_summary.onnx_models.items():
        rows.extend(build_model_metric_rows(model_name, metrics))
    return tuple(rows)


def build_model_metric_rows(model_name: str, metrics: ModelEvaluationMetrics) -> list[tuple[str, ...]]:
    """Return metric rows for one model."""
    rows: list[tuple[str, ...]] = []
    for test_label, evaluation in (
        ("full test", metrics["test_full"]),
        ("screen-balanced test", metrics["test_screen_balanced"]),
        ("safety-balanced test", metrics["test_safety_balanced"]),
    ):
        rows.extend(build_head_metric_rows(model_name, test_label, evaluation))
    return rows


def build_head_metric_rows(model_name: str, test_label: str, metrics: EvaluationMetrics) -> list[tuple[str, ...]]:
    """Return metric rows for both output heads."""
    return [
        (model_name, test_label, head, *format_metric_cells(values))
        for head, values in (("screen", metrics["screen"]), ("safety", metrics["safety"]))
    ]


def format_metric_cells(metrics: HeadMetrics) -> tuple[str, str, str, str, str]:
    """Return formatted metric cells."""
    return (
        format_metric_value(metrics["accuracy"]),
        format_metric_value(metrics["balanced_accuracy"]),
        format_metric_value(metrics["macro_f1"]),
        format_metric_value(metrics["top2_accuracy"]),
        format_metric_value(metrics["total"]),
    )


def collect_timing_values(evaluation_summary: EvaluationSummary) -> tuple[tuple[str, TimingMetrics], ...]:
    """Return full-test timing grouped by model format."""
    values: list[tuple[str, TimingMetrics]] = []
    for model_name, metrics in evaluation_summary.onnx_models.items():
        timing = metrics["test_full"].get("timing")
        if timing is not None:
            values.append((model_name, timing))
    return tuple(values)


def build_timing_table(timings: tuple[tuple[str, TimingMetrics], ...]) -> Table:
    """Build the final timing table."""
    table = Table(expand=True, show_header=True, header_style="bold", title="Test Timing")
    table.add_column("model")
    table.add_column("stage")
    table.add_column("mean", justify="right")
    table.add_column("median", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("reference", justify="right")
    for model_name, timing in timings:
        table.add_row(
            model_name,
            "total",
            *format_timing_cells(timing["total"]),
            f"{timing['images_per_second']:.1f} img/s",
        )
        table.add_row(
            model_name,
            "load/preprocess",
            *format_timing_cells(timing["read_preprocess"]),
            f"{timing['image_count']} images",
        )
        table.add_row(
            model_name,
            "inference",
            *format_timing_cells(timing["model_run"]),
            format_timing_context(timing),
        )
    return table


def format_timing_cells(stats: LatencyStats) -> tuple[str, str, str]:
    """Return formatted timing cells."""
    return (
        format_elapsed_time(stats["mean_ms"]),
        format_elapsed_time(stats["median_ms"]),
        format_elapsed_time(stats["p95_ms"]),
    )


def format_timing_context(timing: TimingMetrics) -> str:
    """Format timing context."""
    return f"{timing['provider']} / batch {timing['batch_size']} / {timing['image_size']}px"


def format_progress_bar(done: int, total: int, width: int = DASHBOARD_PROGRESS_WIDTH) -> str:
    """Format a text progress bar."""
    if total <= 0:
        return "░" * width
    filled = min(width, max(0, round(width * done / total)))
    return "█" * filled + "░" * (width - filled)


def format_sparkline(points: list[float], width: int = DASHBOARD_SPARKLINE_WIDTH) -> str:
    """Format a loss sparkline."""
    if not points:
        return "·" * width
    values = points[-width:]
    low = min(values)
    high = max(values)
    if high == low:
        return DASHBOARD_SPARKLINE_BLOCKS[0] * len(values)
    return "".join(
        DASHBOARD_SPARKLINE_BLOCKS[round((point - low) * (len(DASHBOARD_SPARKLINE_BLOCKS) - 1) / (high - low))]
        for point in values
    )
