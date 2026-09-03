"""Build the local Markdown training report."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.domains.artifacts.files import write_text

if TYPE_CHECKING:
    from pathlib import Path
    from src.state.metrics import (
        LatencyStats,
        TimingMetrics,
        TrainingReport,
        EvaluationMetrics,
    )


def write_report_markdown(path: Path, report: TrainingReport) -> None:
    """Write a readable report from the typed JSON report model."""
    lines = [
        "# Training Report",
        "",
        f"- Model: `{report['model']}`",
        f"- Image size: `{report['image_size']}`",
        f"- Best checkpoint: `{report['best_checkpoint']}`",
    ]
    for heading, metrics in (
        ("Validation Full", report["val_full"]),
        ("Validation Screen-Balanced", report["val_screen_balanced"]),
        ("Validation Safety-Balanced", report["val_safety_balanced"]),
        ("Test Full", report["test_full"]),
        ("Test Screen-Balanced", report["test_screen_balanced"]),
        ("Test Safety-Balanced", report["test_safety_balanced"]),
    ):
        append_metric_section(lines, heading, metrics)
    append_timing_section(lines, report["test_full"].get("timing"))
    write_text(path, "\n".join(lines) + "\n")


def append_metric_section(lines: list[str], heading: str, metrics: EvaluationMetrics) -> None:
    """Append one evaluation section."""
    lines.extend(["", f"## {heading}"])
    for label, values in (("Screen", metrics["screen"]), ("Safety", metrics["safety"])):
        lines.extend(
            [
                "",
                f"### {label}",
                "",
                f"- Macro F1: `{values['macro_f1']:.6f}`",
                f"- Accuracy: `{values['accuracy']:.6f}`",
                f"- Balanced accuracy: `{values['balanced_accuracy']:.6f}`",
                f"- Top-2 accuracy: `{values['top2_accuracy']:.6f}`",
                f"- Images: `{values['total']}`",
            ],
        )


def append_timing_section(lines: list[str], timing: TimingMetrics | None) -> None:
    """Append measured full-test timing when present."""
    if timing is None:
        return
    lines.extend(
        [
            "",
            "## Test Timing",
            "",
            f"- Provider: `{timing['provider']}`",
            f"- Device: `{timing['device']}`",
            f"- Images: `{timing['image_count']}`",
            f"- Image size: `{timing['image_size']}`",
            f"- Batch size: `{timing['batch_size']}`",
            f"- Images per second: `{timing['images_per_second']:.6f}`",
            timing_stat_line("Load/preprocess", timing["read_preprocess"]),
            timing_stat_line("Model", timing["model_run"]),
            timing_stat_line("Total", timing["total"]),
        ],
    )


def timing_stat_line(label: str, stats: LatencyStats) -> str:
    """Format one latency distribution line."""
    return (
        f"- {label}: mean `{stats['mean_ms']:.6f} ms`, median `{stats['median_ms']:.6f} ms`, "
        f"p95 `{stats['p95_ms']:.6f} ms`"
    )
