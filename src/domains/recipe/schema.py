"""Validate persisted training recipes."""

from __future__ import annotations

import math
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
from src.config.model import TRAINABLE_SAFETY_LABELS
from src.state.recipe import (
    LossRecipe,
    CountRecipe,
    LabelRecipe,
    DatasetRecipe,
    SamplingRecipe,
    TrainingRecipe,
    OptimizerRecipe,
    EvaluationRecipe,
    PreprocessRecipe,
    TrainingRunRecipe,
    AugmentationRecipe,
    TaskScheduleRecipe,
    CheckpointScoreRecipe,
)

if TYPE_CHECKING:
    from typing import Never

NORMALIZATION_VALUE_COUNT = 3
TRAINING_RECIPE_SCHEMA = 4


def _error(detail: str) -> Never:
    """Raise one stable recipe validation error."""
    message = f"Saved training recipe is invalid: {detail}."
    raise TrainingError(message)


def _record(value: object, name: str, fields: set[str]) -> dict[str, object]:
    """Require one exact string-keyed object."""
    if not isinstance(value, dict):
        _error(f"{name} must be an object")
    values = cast("dict[object, object]", value)
    if set(values) != fields or any(not isinstance(key, str) for key in values):
        _error(f"{name} fields do not match the schema")
    return cast("dict[str, object]", values)


def _string(value: object, name: str) -> str:
    """Require one non-empty string."""
    if not isinstance(value, str) or not value:
        _error(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Require one bounded non-Boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(f"{name} must be an integer of at least {minimum}")
    return value


def _number(value: object, name: str) -> float:
    """Require one finite non-Boolean number."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        _error(f"{name} must be a finite number")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    """Require one Boolean value."""
    if not isinstance(value, bool):
        _error(f"{name} must be Boolean")
    return value


def _strings(value: object, name: str) -> list[str]:
    """Require a non-empty list of unique non-empty strings."""
    if not isinstance(value, list):
        _error(f"{name} must be a string list")
    values = cast("list[object]", value)
    if not values or any(not isinstance(item, str) or not item for item in values):
        _error(f"{name} must be a string list")
    strings = cast("list[str]", values)
    if len(strings) != len(set(strings)):
        _error(f"{name} must not contain duplicates")
    return strings


def _normalization(value: object, name: str) -> list[float]:
    """Require exactly three finite normalization values."""
    if not isinstance(value, list):
        _error(f"{name} must contain three values")
    values = cast("list[object]", value)
    if len(values) != NORMALIZATION_VALUE_COUNT:
        _error(f"{name} must contain three values")
    return [_number(item, name) for item in values]


def _string_map(value: object, name: str) -> dict[str, str]:
    """Require a string-to-string object."""
    if not isinstance(value, dict):
        _error(f"{name} must be an object")
    values = cast("dict[object, object]", value)
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in values.items()):
        _error(f"{name} must map strings to strings")
    return cast("dict[str, str]", values)


def _count_map(value: object, name: str) -> dict[str, int]:
    """Require a string-to-nonnegative-integer object."""
    if not isinstance(value, dict):
        _error(f"{name} must be an object")
    values = cast("dict[object, object]", value)
    if any(
        not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, int) or item < 0
        for key, item in values.items()
    ):
        _error(f"{name} must contain non-negative integer counts")
    return cast("dict[str, int]", values)


def _dataset(value: object) -> DatasetRecipe:
    """Validate the dataset section."""
    values = _record(value, "dataset", {"path", "repo", "safety_field", "screen_field"})
    screen_field = _string(values["screen_field"], "dataset.screen_field")
    safety_field = _string(values["safety_field"], "dataset.safety_field")
    if screen_field != "screen" or safety_field != "safety":
        _error("dataset label fields are unsupported")
    return DatasetRecipe(
        repo=_string(values["repo"], "dataset.repo"),
        path=_string(values["path"], "dataset.path"),
        screen_field="screen",
        safety_field="safety",
    )


def _labels(value: object) -> LabelRecipe:
    """Validate the label section."""
    values = _record(value, "labels", {"safety", "screen"})
    safety = _strings(values["safety"], "labels.safety")
    unsupported = sorted(set(safety) - set(TRAINABLE_SAFETY_LABELS))
    if unsupported:
        _error(f"labels.safety contains unsupported training labels: {', '.join(unsupported)}")
    return LabelRecipe(
        screen=_strings(values["screen"], "labels.screen"),
        safety=safety,
    )


def _preprocess(value: object) -> PreprocessRecipe:
    """Validate the preprocessing section."""
    fields = {
        "crop",
        "horizontal_flip",
        "mean",
        "preserve_aspect_ratio",
        "resize_longest_side_px",
        "std",
        "stretch",
    }
    values = _record(value, "preprocess", fields)
    return PreprocessRecipe(
        resize_longest_side_px=_integer(values["resize_longest_side_px"], "preprocess.resize", minimum=1),
        preserve_aspect_ratio=_boolean(values["preserve_aspect_ratio"], "preprocess.preserve_aspect_ratio"),
        crop=_boolean(values["crop"], "preprocess.crop"),
        stretch=_boolean(values["stretch"], "preprocess.stretch"),
        horizontal_flip=_boolean(values["horizontal_flip"], "preprocess.horizontal_flip"),
        mean=_normalization(values["mean"], "preprocess.mean"),
        std=_normalization(values["std"], "preprocess.std"),
    )


def _augmentation(value: object) -> AugmentationRecipe:
    """Validate the augmentation section."""
    fields = {
        "artifact_codecs",
        "artifact_probability",
        "artifact_quality_max",
        "artifact_quality_min",
        "brightness_max",
        "brightness_min",
        "brightness_probability",
        "contrast_max",
        "contrast_min",
        "contrast_probability",
    }
    values = _record(value, "augmentation", fields)
    return AugmentationRecipe(
        brightness_probability=_number(values["brightness_probability"], "augmentation.brightness_probability"),
        brightness_min=_number(values["brightness_min"], "augmentation.brightness_min"),
        brightness_max=_number(values["brightness_max"], "augmentation.brightness_max"),
        contrast_probability=_number(values["contrast_probability"], "augmentation.contrast_probability"),
        contrast_min=_number(values["contrast_min"], "augmentation.contrast_min"),
        contrast_max=_number(values["contrast_max"], "augmentation.contrast_max"),
        artifact_probability=_number(values["artifact_probability"], "augmentation.artifact_probability"),
        artifact_quality_min=_integer(values["artifact_quality_min"], "augmentation.artifact_quality_min"),
        artifact_quality_max=_integer(values["artifact_quality_max"], "augmentation.artifact_quality_max"),
        artifact_codecs=_strings(values["artifact_codecs"], "augmentation.artifact_codecs"),
    )


def _training(value: object) -> TrainingRunRecipe:
    """Validate epoch and iterator settings."""
    fields = {
        "batch_size",
        "effective_batch_size",
        "epochs",
        "grad_accum_steps",
        "seed",
        "target_effective_batch_size",
        "workers",
    }
    values = _record(value, "training", fields)
    return TrainingRunRecipe(
        epochs=_integer(values["epochs"], "training.epochs", minimum=1),
        batch_size=_integer(values["batch_size"], "training.batch_size", minimum=1),
        workers=_integer(values["workers"], "training.workers"),
        seed=_integer(values["seed"], "training.seed"),
        grad_accum_steps=_integer(values["grad_accum_steps"], "training.grad_accum_steps", minimum=1),
        effective_batch_size=_integer(values["effective_batch_size"], "training.effective_batch_size", minimum=1),
        target_effective_batch_size=_integer(
            values["target_effective_batch_size"],
            "training.target_effective_batch_size",
            minimum=1,
        ),
    )


def _optimizer(value: object) -> OptimizerRecipe:
    """Validate optimizer settings."""
    values = _record(value, "optimizer", {"backbone_lr", "head_lr", "lr", "weight_decay"})
    return OptimizerRecipe(
        lr=_number(values["lr"], "optimizer.lr"),
        backbone_lr=_number(values["backbone_lr"], "optimizer.backbone_lr"),
        head_lr=_number(values["head_lr"], "optimizer.head_lr"),
        weight_decay=_number(values["weight_decay"], "optimizer.weight_decay"),
    )


def _sampling(value: object) -> SamplingRecipe:
    """Validate sampling settings."""
    fields = {
        "folded_screen_labels",
        "min_train_count",
        "mode",
        "safety_max_repeat",
        "safety_target_ratio",
        "screen_max_repeat",
        "screen_target_ratio",
    }
    values = _record(value, "sampling", fields)
    if values["mode"] != "adaptive_label_aspect_bucketed":
        _error("sampling.mode is unsupported")
    return SamplingRecipe(
        mode="adaptive_label_aspect_bucketed",
        min_train_count=_integer(values["min_train_count"], "sampling.min_train_count", minimum=1),
        screen_target_ratio=_number(values["screen_target_ratio"], "sampling.screen_target_ratio"),
        safety_target_ratio=_number(values["safety_target_ratio"], "sampling.safety_target_ratio"),
        screen_max_repeat=_integer(values["screen_max_repeat"], "sampling.screen_max_repeat", minimum=1),
        safety_max_repeat=_integer(values["safety_max_repeat"], "sampling.safety_max_repeat", minimum=1),
        folded_screen_labels=_string_map(values["folded_screen_labels"], "sampling.folded_screen_labels"),
    )


def _loss(value: object) -> LossRecipe:
    """Validate loss and task schedule settings."""
    values = _record(
        value,
        "loss",
        {"safety", "safety_batch_probability", "safety_loss_weight", "screen", "task_schedule"},
    )
    schedule = _record(
        values["task_schedule"],
        "loss.task_schedule",
        {"policy", "safety_batch_losses", "screen_batch_losses", "safety_training_batches"},
    )
    if (
        values["screen"] != "cross_entropy"
        or values["safety"] != "cross_entropy"
        or schedule["policy"] != "deterministic_cover_both_loaders"
        or schedule["screen_batch_losses"] != ["screen", "safety"]
        or schedule["safety_batch_losses"] != ["safety"]
        or schedule["safety_training_batches"] is not True
    ):
        _error("loss task policy is unsupported")
    return LossRecipe(
        screen="cross_entropy",
        safety="cross_entropy",
        safety_loss_weight=_number(values["safety_loss_weight"], "loss.safety_loss_weight"),
        safety_batch_probability=_number(values["safety_batch_probability"], "loss.safety_batch_probability"),
        task_schedule=TaskScheduleRecipe(
            policy="deterministic_cover_both_loaders",
            screen_batch_losses=["screen", "safety"],
            safety_batch_losses=["safety"],
            safety_training_batches=True,
        ),
    )


def _evaluation(value: object) -> EvaluationRecipe:
    """Validate evaluation and checkpoint scoring settings."""
    values = _record(value, "evaluation", {"eval_class_limit", "checkpoint_score"})
    score = _record(values["checkpoint_score"], "evaluation.checkpoint_score", {"accuracy_weight", "formula"})
    return EvaluationRecipe(
        eval_class_limit=_integer(
            values["eval_class_limit"],
            "evaluation.eval_class_limit",
            minimum=1,
        ),
        checkpoint_score=CheckpointScoreRecipe(
            accuracy_weight=_number(score["accuracy_weight"], "evaluation.checkpoint_score.accuracy_weight"),
            formula=_string(score["formula"], "evaluation.checkpoint_score.formula"),
        ),
    )


def _counts(value: object) -> CountRecipe:
    """Validate split count records."""
    fields = {"test_safety", "test_screen", "train_safety", "train_screen", "val_safety", "val_screen"}
    values = _record(value, "counts", fields)
    return CountRecipe(
        train_screen=_count_map(values["train_screen"], "counts.train_screen"),
        val_screen=_count_map(values["val_screen"], "counts.val_screen"),
        test_screen=_count_map(values["test_screen"], "counts.test_screen"),
        train_safety=_count_map(values["train_safety"], "counts.train_safety"),
        val_safety=_count_map(values["val_safety"], "counts.val_safety"),
        test_safety=_count_map(values["test_safety"], "counts.test_safety"),
    )


def parse_training_recipe(value: object) -> TrainingRecipe:
    """Validate and return a complete version-four training recipe."""
    fields = {
        "augmentation",
        "counts",
        "dataset",
        "evaluation",
        "labels",
        "loss",
        "model",
        "optimizer",
        "preprocess",
        "sampling",
        "schema",
        "training",
    }
    values = _record(value, "root", fields)
    if values["schema"] != TRAINING_RECIPE_SCHEMA:
        _error("schema version is unsupported")
    return TrainingRecipe(
        schema=4,
        dataset=_dataset(values["dataset"]),
        model=_string(values["model"], "model"),
        labels=_labels(values["labels"]),
        preprocess=_preprocess(values["preprocess"]),
        augmentation=_augmentation(values["augmentation"]),
        training=_training(values["training"]),
        optimizer=_optimizer(values["optimizer"]),
        sampling=_sampling(values["sampling"]),
        loss=_loss(values["loss"]),
        evaluation=_evaluation(values["evaluation"]),
        counts=_counts(values["counts"]),
    )
