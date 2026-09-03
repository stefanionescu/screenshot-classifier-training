"""Build the exported model card from typed evaluation metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import UTC, datetime
from src.domains.train.artifacts.files import write_text
from src.config.train.artifacts import (
    TRAIN_ONNX_DIR,
    TRAIN_MODEL_NAME,
    ONNX_OUTPUT_NAMES,
    ONNX_MODEL_FILENAME,
    TRAIN_INFERENCE_DIR,
    TRAIN_MODEL_LIBRARY,
    TRAIN_TEXT_ENCODING,
    TRAIN_CHECKPOINTS_DIR,
    TRAIN_CONFIG_FILENAME,
    TRAIN_LABELS_FILENAME,
    TRAIN_README_FILENAME,
    ONNX_HALF_MODEL_FILENAME,
    TRAIN_PREPROCESS_FILENAME,
    TRAIN_MODEL_CONFIG_FILENAME,
    TRAIN_MODEL_WEIGHTS_FILENAME,
    TRAIN_PYTHON_INFERENCE_FILENAME,
    TRAIN_EXPORT_CHECKPOINT_FILENAME,
)

if TYPE_CHECKING:
    from src.state.train.export import ModelCard
    from src.state.train.metrics import (
        HeadMetrics,
        TimingMetrics,
        EvaluationMetrics,
        EvaluationSummary,
        ModelEvaluationMetrics,
    )

MODEL_CARD_SOURCE = Path(__file__).with_name("model_card.md")
PLACEHOLDER_PATTERN = re.compile(r"{{[a-z_]+}}")
RESULT_HEADERS = (
    "Model",
    "Test",
    "Output",
    "Accuracy",
    "Balanced accuracy",
    "Macro F1",
    "Top-2",
    "Images",
)
TIMING_HEADERS = (
    "Model",
    "Images/s",
    "Load Mean",
    "Model Mean",
    "Total Mean",
    "Total Median",
    "Total P95",
    "Provider",
)
RESULTS_PENDING = "Final test metrics are written after evaluation."
RESULTS_UNAVAILABLE = "Final test metrics are unavailable."
TIMING_PENDING = "ONNX CPU timing is written after evaluation."
TIMING_UNAVAILABLE = "ONNX CPU timing is unavailable."
HARDWARE_PENDING = "the benchmark CPU"


def write_model_card(card: ModelCard) -> None:
    """Write a fully resolved model card."""
    content = build_model_card(card.model_id, card.repo_id, card.report)
    write_text(card.export_dir / TRAIN_README_FILENAME, content)


def build_model_card(
    model_id: str,
    repo_id: str,
    report: EvaluationSummary | None = None,
) -> str:
    """Build the model-card template and reject unresolved placeholders."""
    artifacts = (
        (f"{TRAIN_ONNX_DIR}/{ONNX_MODEL_FILENAME}", "ONNX model for inference."),
        (
            f"{TRAIN_ONNX_DIR}/{ONNX_HALF_MODEL_FILENAME}",
            "Optional half-precision ONNX model.",
        ),
        (
            f"{TRAIN_ONNX_DIR}/{ONNX_MODEL_FILENAME}.data",
            "External ONNX tensor data.",
        ),
        (TRAIN_MODEL_WEIGHTS_FILENAME, "PyTorch weights for continued training."),
        (TRAIN_MODEL_CONFIG_FILENAME, "Model identity, outputs, and label arrays."),
        (TRAIN_PREPROCESS_FILENAME, "Image preprocessing contract."),
        (TRAIN_CONFIG_FILENAME, "Training settings used for this export."),
        (
            f"{TRAIN_CHECKPOINTS_DIR}/{TRAIN_EXPORT_CHECKPOINT_FILENAME}",
            "Selected training checkpoint.",
        ),
        (
            f"{TRAIN_INFERENCE_DIR}/{TRAIN_PYTHON_INFERENCE_FILENAME}",
            "Standalone Python inference helper.",
        ),
        (
            f"{TRAIN_INFERENCE_DIR}/{TRAIN_LABELS_FILENAME}",
            "Labels used to decode logits.",
        ),
        (TRAIN_README_FILENAME, "This generated model card."),
    )
    replacements = {
        "artifact_table": "\n".join(f"| `{path}` | {purpose} |" for path, purpose in artifacts),
        "citation_year": str(datetime.now(UTC).year),
        "model_id": model_id,
        "library_name": TRAIN_MODEL_LIBRARY,
        "model_name": TRAIN_MODEL_NAME,
        "output_names": " and ".join(f"`{name}`" for name in ONNX_OUTPUT_NAMES),
        "repo_id": repo_id,
        "test_results": test_results_section(report),
        "test_timing": test_timing_section(report),
        "benchmark_hardware": benchmark_hardware_label(report),
    }
    content = MODEL_CARD_SOURCE.read_text(encoding=TRAIN_TEXT_ENCODING)
    for name, value in replacements.items():
        content = content.replace(f"{{{{{name}}}}}", value)
    unresolved = sorted(set(PLACEHOLDER_PATTERN.findall(content)))
    if unresolved:
        msg = f"model card has unresolved placeholders: {', '.join(unresolved)}"
        raise ValueError(msg)
    return content if content.endswith("\n") else f"{content}\n"


def test_results_section(report: EvaluationSummary | None) -> str:
    """Build the card result section."""
    if report is None:
        return RESULTS_PENDING
    rows = [row for model_name, metrics in report.onnx_models.items() for row in model_result_rows(model_name, metrics)]
    return markdown_table(RESULT_HEADERS, rows) if rows else RESULTS_UNAVAILABLE


def test_timing_section(report: EvaluationSummary | None) -> str:
    """Build the card timing section."""
    if report is None:
        return TIMING_PENDING
    rows = [
        timing_row(model_name, timing)
        for model_name, metrics in report.onnx_models.items()
        if (timing := metrics["test_full"].get("timing")) is not None
    ]
    return markdown_table(TIMING_HEADERS, rows) if rows else TIMING_UNAVAILABLE


def model_result_rows(model_name: str, metrics: ModelEvaluationMetrics) -> list[tuple[str, ...]]:
    """Return result rows for one exported model."""
    rows: list[tuple[str, ...]] = []
    for test_label, evaluation in (
        ("full test", metrics["test_full"]),
        ("screen-balanced test", metrics["test_screen_balanced"]),
        ("safety-balanced test", metrics["test_safety_balanced"]),
    ):
        rows.extend(test_set_result_rows(model_name, test_label, evaluation))
    return rows


def test_set_result_rows(model_name: str, test_label: str, metrics: EvaluationMetrics) -> list[tuple[str, ...]]:
    """Return result rows for both model outputs."""
    return [
        test_result_row(model_name, test_label, output_name, head_metrics)
        for output_name, head_metrics in (("screen", metrics["screen"]), ("safety", metrics["safety"]))
    ]


def test_result_row(model_name: str, test_label: str, output_name: str, metrics: HeadMetrics) -> tuple[str, ...]:
    """Return one model-card result row."""
    return (
        model_name,
        test_label,
        output_name,
        metric(metrics["accuracy"]),
        metric(metrics["balanced_accuracy"]),
        metric(metrics["macro_f1"]),
        metric(metrics["top2_accuracy"]),
        str(metrics["total"]),
    )


def timing_row(label: str, timing: TimingMetrics) -> tuple[str, ...]:
    """Return one model-card timing row."""
    reading = timing["read_preprocess"]
    model = timing["model_run"]
    total = timing["total"]
    return (
        label,
        metric(timing["images_per_second"]),
        milliseconds(reading["mean_ms"]),
        milliseconds(model["mean_ms"]),
        milliseconds(total["mean_ms"]),
        milliseconds(total["median_ms"]),
        milliseconds(total["p95_ms"]),
        timing["provider"],
    )


def benchmark_hardware_label(report: EvaluationSummary | None) -> str:
    """Return the benchmark hardware label."""
    if report is None:
        return HARDWARE_PENDING
    timing = report.test["test_full"].get("timing")
    return timing.get("hardware", HARDWARE_PENDING) if timing is not None else HARDWARE_PENDING


def markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Build a Markdown table."""
    alignment = tuple(":--" if index == 0 else "--:" for index in range(len(headers)))
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def metric(value: float) -> str:
    """Format a metric value."""
    return f"{value:.4f}"


def milliseconds(value: float) -> str:
    """Format milliseconds."""
    return f"{value:.2f} ms"
