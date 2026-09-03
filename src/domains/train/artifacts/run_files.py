"""Canonical paths and append operations within a training run."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from src.shared.output import json_safe
from src.config.train.artifacts import (
    TRAIN_LOCAL_DIR,
    VAL_ERRORS_FILENAME,
    TEST_ERRORS_FILENAME,
    TRAIN_LOCAL_LOGS_DIR,
    TRAIN_CONFIG_FILENAME,
    TRAIN_REPORT_FILENAME,
    TRAIN_LOCAL_CONFIG_DIR,
    TRAIN_LOCAL_METRICS_DIR,
    TOP_TWO_RESCUES_FILENAME,
    TRAIN_LOCAL_ANALYSIS_DIR,
    TRAIN_LOCAL_FAILURES_DIR,
    TRAIN_LOCAL_PREDICTIONS_DIR,
    TRAIN_SKIPPED_IMAGES_FILENAME,
    CONFIDENT_WRONG_EIGHTY_FILENAME,
    CONFIDENT_WRONG_NINETY_FILENAME,
    TRAIN_UNCONFIDENT_WRONG_FILENAME,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterable


def local_train_dir(run_dir: Path) -> Path:
    """Return the private training-state directory for a run."""
    return run_dir / TRAIN_LOCAL_DIR


def config_dir(run_dir: Path) -> Path:
    """Return the run directory containing resolved configuration."""
    return local_train_dir(run_dir) / TRAIN_LOCAL_CONFIG_DIR


def metrics_dir(run_dir: Path) -> Path:
    """Return the run directory containing metric reports."""
    return local_train_dir(run_dir) / TRAIN_LOCAL_METRICS_DIR


def predictions_dir(run_dir: Path) -> Path:
    """Return the run directory containing prediction journals."""
    return local_train_dir(run_dir) / TRAIN_LOCAL_PREDICTIONS_DIR


def failures_dir(run_dir: Path) -> Path:
    """Return the run directory containing diagnostic failure sets."""
    return local_train_dir(run_dir) / TRAIN_LOCAL_FAILURES_DIR


def analysis_dir(run_dir: Path) -> Path:
    """Return the run directory containing derived analyses."""
    return local_train_dir(run_dir) / TRAIN_LOCAL_ANALYSIS_DIR


def logs_dir(run_dir: Path) -> Path:
    """Return the run directory containing operational logs."""
    return local_train_dir(run_dir) / TRAIN_LOCAL_LOGS_DIR


def train_config_path(run_dir: Path) -> Path:
    """Train config path."""
    return config_dir(run_dir) / TRAIN_CONFIG_FILENAME


def skipped_images_path(run_dir: Path) -> Path:
    """Return the journal path for images rejected during decoding."""
    return logs_dir(run_dir) / TRAIN_SKIPPED_IMAGES_FILENAME


def report_json_path(run_dir: Path) -> Path:
    """Return the machine-readable training report path."""
    return metrics_dir(run_dir) / TRAIN_REPORT_FILENAME


def prediction_path(run_dir: Path, stage: str) -> Path:
    """Return the prediction journal path for an evaluation stage."""
    return predictions_dir(run_dir) / f"{stage}.jsonl"


def failure_paths(run_dir: Path) -> dict[str, Path]:
    """Return every named diagnostic failure-set path for a run."""
    directory = failures_dir(run_dir)
    return {
        "val_full_errors": directory / VAL_ERRORS_FILENAME,
        "test_full_errors": directory / TEST_ERRORS_FILENAME,
        "top2_rescues": directory / TOP_TWO_RESCUES_FILENAME,
        "confident_wrong_80": directory / CONFIDENT_WRONG_EIGHTY_FILENAME,
        "confident_wrong_90": directory / CONFIDENT_WRONG_NINETY_FILENAME,
        "unconfident_wrong": directory / TRAIN_UNCONFIDENT_WRONG_FILENAME,
    }


def prepare_local_artifact_dirs(run_dir: Path) -> None:
    """Prepare local artifact dirs."""
    for directory in (
        config_dir(run_dir),
        metrics_dir(run_dir),
        predictions_dir(run_dir),
        failures_dir(run_dir),
        analysis_dir(run_dir),
        logs_dir(run_dir),
    ):
        directory.mkdir(parents=True, exist_ok=True)


def reset_file(path: Path) -> None:
    """Create or truncate an append-only run artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_jsonl(path: Path, row: object) -> None:
    """Validate and append one JSONL row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def append_jsonl_many(path: Path, rows: Iterable[object]) -> None:
    """Validate and append several JSONL rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
