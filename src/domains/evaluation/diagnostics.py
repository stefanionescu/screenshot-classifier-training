"""Persist predictions and derive failure and calibration diagnostics."""

from __future__ import annotations

import json
import math
import torch
import itertools
import numpy as np
from pathlib import Path
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
from src.domains.artifacts.paths import repo_root
from src.domains.artifacts.files import write_json
from src.config.artifacts import TRAIN_CALIBRATION_FILENAME
from src.state.eval import CalibrationReport, CalibrationThresholds, ConfidenceBucket, PredictionRow
from src.config.metrics import (
    CONFIDENT_SCORE,
    PROBABILITY_FLOOR,
    CONFIDENCE_BUCKETS,
    LOW_CONFIDENCE_SCORE,
    VERY_CONFIDENT_SCORE,
)
from src.domains.artifacts.runs import (
    reset_file,
    analysis_dir,
    append_jsonl,
    failure_paths,
    prediction_path,
    append_jsonl_many,
)

if TYPE_CHECKING:
    from typing import Literal
    from collections.abc import Iterable
    from src.state.contracts import DatasetSplit
    from src.state.eval import EvalArtifacts, PredBatch


def reset_eval_artifacts(run_dir: Path, stages: Iterable[str]) -> None:
    """Reset prediction and failure journals before a fresh evaluation."""
    for stage in stages:
        reset_file(prediction_path(run_dir, stage))
    for path in failure_paths(run_dir).values():
        reset_file(path)


def prediction_rows(
    artifacts: EvalArtifacts,
    batch: PredBatch,
) -> list[PredictionRow]:
    """Build persisted prediction records for one evaluated batch."""
    screen_log_probs, screen_probs = log_probabilities(batch.screen_logits)
    safety_log_probs, safety_probs = log_probabilities(batch.safety_logits)
    screen_target_ids = to_numpy(batch.screen_targets).astype(int)
    safety_target_ids = to_numpy(batch.safety_targets).astype(int)
    rows: list[PredictionRow] = []
    for index, meta in enumerate(batch.metas):
        screen_order = list(np.argsort(screen_probs[index])[::-1])
        safety_order = list(np.argsort(safety_probs[index])[::-1])
        screen_pred = int(screen_order[0])
        safety_pred = int(safety_order[0])
        screen_target = int(screen_target_ids[index])
        safety_target = int(safety_target_ids[index])
        screen_top2 = screen_order[: min(2, len(batch.screen_labels))]
        rows.append(
            PredictionRow(
                stage=artifacts.stage,
                split=artifacts.split,
                tar_path=relative_path(meta.tar_path),
                image_member=meta.image_member,
                width=meta.width,
                height=meta.height,
                true_screen=batch.screen_labels[screen_target],
                pred_screen=batch.screen_labels[screen_pred],
                screen_correct=screen_pred == screen_target,
                screen_top2=[batch.screen_labels[int(item)] for item in screen_top2],
                screen_top2_hit=screen_target in screen_top2,
                screen_confidence=float(screen_probs[index, screen_pred]),
                true_safety=safety_label(batch.safety_labels, safety_target),
                pred_safety=batch.safety_labels[safety_pred],
                safety_correct=(
                    safety_pred == safety_target if is_active_label(batch.safety_labels, safety_target) else None
                ),
                safety_confidence=float(safety_probs[index, safety_pred]),
                screen_scores=label_scores(batch.screen_labels, screen_log_probs[index]),
                safety_scores=label_scores(batch.safety_labels, safety_log_probs[index]),
            ),
        )
    return rows


def safety_label(labels: list[str], target: int) -> str | None:
    """Return an active safety label or no label for an ignored target."""
    if not is_active_label(labels, target):
        return None
    return labels[target]


def is_active_label(labels: list[str], target: int) -> bool:
    """Return whether a target identifies a configured label."""
    return 0 <= target < len(labels)


def write_prediction_rows(artifacts: EvalArtifacts, rows: list[PredictionRow]) -> None:
    """Write prediction rows."""
    if rows:
        append_jsonl_many(prediction_path(artifacts.run_dir, artifacts.stage), rows)


def write_failures_from_rows(run_dir: Path, rows: list[PredictionRow]) -> None:
    """Write failures from rows."""
    paths = failure_paths(run_dir)
    for row in rows:
        screen_wrong = row["screen_correct"] is False
        safety_wrong = row["safety_correct"] is False
        screen_conf = row["screen_confidence"]
        safety_conf = row["safety_confidence"]
        if row["stage"] == "val_full" and (screen_wrong or safety_wrong):
            append_jsonl(paths["val_full_errors"], row)
        if row["stage"] == "test_full" and (screen_wrong or safety_wrong):
            append_jsonl(paths["test_full_errors"], row)
        if screen_wrong and row["screen_top2_hit"] is True:
            append_jsonl(paths["top2_rescues"], row)
        if (screen_wrong and screen_conf >= CONFIDENT_SCORE) or (safety_wrong and safety_conf >= CONFIDENT_SCORE):
            append_jsonl(paths["confident_wrong_80"], row)
        if (screen_wrong and screen_conf >= VERY_CONFIDENT_SCORE) or (
            safety_wrong and safety_conf >= VERY_CONFIDENT_SCORE
        ):
            append_jsonl(paths["confident_wrong_90"], row)
        if (screen_wrong and screen_conf < LOW_CONFIDENCE_SCORE) or (
            safety_wrong and safety_conf < LOW_CONFIDENCE_SCORE
        ):
            append_jsonl(paths["unconfident_wrong"], row)


def _prediction_record(value: object) -> dict[str, object]:
    """Require one exact string-keyed prediction object."""
    if not isinstance(value, dict):
        msg = "prediction journal contains a non-object record."
        raise TrainingError(msg)
    values = cast("dict[object, object]", value)
    expected = set(PredictionRow.__required_keys__)
    if set(values) != expected or any(not isinstance(key, str) for key in values):
        msg = "prediction journal fields do not match the supported schema."
        raise TrainingError(msg)
    return cast("dict[str, object]", values)


def _prediction_string(value: object, field: str) -> str:
    """Require one non-empty prediction string."""
    if not isinstance(value, str) or not value:
        msg = f"prediction journal {field} must be a non-empty string."
        raise TrainingError(msg)
    return value


def _prediction_number(value: object, field: str) -> float:
    """Require one finite prediction number."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        msg = f"prediction journal {field} must be finite."
        raise TrainingError(msg)
    return float(value)


def _prediction_dimension(value: object, field: str) -> int:
    """Require one positive prediction image dimension."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"prediction journal {field} must be a positive integer."
        raise TrainingError(msg)
    return value


def _prediction_scores(value: object, field: str) -> dict[str, float]:
    """Require one non-empty map of finite label scores."""
    if not isinstance(value, dict):
        msg = f"prediction journal {field} must be a score object."
        raise TrainingError(msg)
    values = cast("dict[object, object]", value)
    if not values or any(not isinstance(label, str) or not label for label in values):
        msg = f"prediction journal {field} labels are invalid."
        raise TrainingError(msg)
    return {cast("str", label): _prediction_number(score, f"{field}.{label}") for label, score in values.items()}


def parse_prediction_row(line: str) -> PredictionRow:
    """Decode and validate one persisted prediction record."""
    try:
        values = _prediction_record(json.loads(line))
    except json.JSONDecodeError as error:
        msg = "prediction journal contains invalid JSON."
        raise TrainingError(msg) from error
    split = _prediction_string(values["split"], "split")
    if split not in {"train", "val", "test"}:
        msg = "prediction journal split is unsupported."
        raise TrainingError(msg)
    top2 = values["screen_top2"]
    if not isinstance(top2, list):
        msg = "prediction journal screen_top2 must be a string list."
        raise TrainingError(msg)
    top2_values = cast("list[object]", top2)
    if any(not isinstance(label, str) or not label for label in top2_values):
        msg = "prediction journal screen_top2 must be a string list."
        raise TrainingError(msg)
    safety_correct = values["safety_correct"]
    true_safety = values["true_safety"]
    if safety_correct is not None and not isinstance(safety_correct, bool):
        msg = "prediction journal safety_correct must be Boolean or null."
        raise TrainingError(msg)
    if true_safety is not None and (not isinstance(true_safety, str) or not true_safety):
        msg = "prediction journal true_safety must be a string or null."
        raise TrainingError(msg)
    return PredictionRow(
        stage=_prediction_string(values["stage"], "stage"),
        split=cast("DatasetSplit", split),
        tar_path=_prediction_string(values["tar_path"], "tar_path"),
        image_member=_prediction_string(values["image_member"], "image_member"),
        width=_prediction_dimension(values["width"], "width"),
        height=_prediction_dimension(values["height"], "height"),
        true_screen=_prediction_string(values["true_screen"], "true_screen"),
        pred_screen=_prediction_string(values["pred_screen"], "pred_screen"),
        screen_correct=_prediction_boolean(values["screen_correct"], "screen_correct"),
        screen_top2=cast("list[str]", top2_values),
        screen_top2_hit=_prediction_boolean(values["screen_top2_hit"], "screen_top2_hit"),
        screen_confidence=_prediction_number(values["screen_confidence"], "screen_confidence"),
        true_safety=true_safety,
        pred_safety=_prediction_string(values["pred_safety"], "pred_safety"),
        safety_correct=safety_correct,
        safety_confidence=_prediction_number(values["safety_confidence"], "safety_confidence"),
        screen_scores=_prediction_scores(values["screen_scores"], "screen_scores"),
        safety_scores=_prediction_scores(values["safety_scores"], "safety_scores"),
    )


def _prediction_boolean(value: object, field: str) -> bool:
    """Require one prediction Boolean."""
    if not isinstance(value, bool):
        msg = f"prediction journal {field} must be Boolean."
        raise TrainingError(msg)
    return value


def write_prediction_analysis(run_dir: Path, stages: Iterable[str]) -> None:
    """Write analysis from prediction files."""
    rows: list[PredictionRow] = []
    for stage in stages:
        path = prediction_path(run_dir, stage)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rows.append(parse_prediction_row(line))
    write_json(analysis_dir(run_dir) / TRAIN_CALIBRATION_FILENAME, calibration_report(rows))


def calibration_report(rows: list[PredictionRow]) -> CalibrationReport:
    """Build screen and safety confidence calibration summaries."""
    return CalibrationReport(
        screen=confidence_buckets(rows, "screen"),
        safety=confidence_buckets(rows, "safety"),
        thresholds=CalibrationThresholds(
            confident_80=CONFIDENT_SCORE,
            confident_90=VERY_CONFIDENT_SCORE,
            unconfident=LOW_CONFIDENCE_SCORE,
        ),
    )


def confidence_buckets(rows: list[PredictionRow], head: Literal["screen", "safety"]) -> list[ConfidenceBucket]:
    """Build calibration buckets for one output head."""
    result: list[ConfidenceBucket] = []
    for lower, upper in itertools.pairwise(CONFIDENCE_BUCKETS):
        bucket = [row for row in rows if in_confidence_bucket(row, head, lower, upper)]
        total = len(bucket)
        correct = sum(1 for row in bucket if head_correct(row, head) is True)
        result.append(
            ConfidenceBucket(
                min=lower,
                max=min(upper, 1.0),
                count=total,
                accuracy=correct / total if total else 0.0,
                wrong_count=total - correct,
                wrong_rate=(total - correct) / total if total else 0.0,
            ),
        )
    return result


def head_correct(row: PredictionRow, head: Literal["screen", "safety"]) -> bool | None:
    """Return the correctness flag for one output head."""
    return row["screen_correct"] if head == "screen" else row["safety_correct"]


def in_confidence_bucket(
    row: PredictionRow,
    head: Literal["screen", "safety"],
    lower: float,
    upper: float,
) -> bool:
    """Return whether one active prediction belongs to a confidence interval."""
    confidence = row["screen_confidence"] if head == "screen" else row["safety_confidence"]
    return head_correct(row, head) is not None and lower <= confidence < upper


def log_probabilities(logits: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert logits into stable probabilities and log probabilities."""
    array = to_numpy(logits).astype(np.float64)
    shifted = array - array.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return np.log(np.maximum(probs, PROBABILITY_FLOOR)), probs


def to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Move a tensor to CPU NumPy storage or preserve an existing array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def label_scores(labels: list[str], values: np.ndarray) -> dict[str, float]:
    """Map labels to finite log-probability scores."""
    return {label: float(values[index]) for index, label in enumerate(labels)}


def relative_path(path: str) -> str:
    """Use a project-relative artifact path when the source is in-tree."""
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(repo_root()))
    except ValueError:
        return str(resolved)
