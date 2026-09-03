"""Typed persisted training recipe schema."""

from __future__ import annotations

from typing import Literal, TypedDict


class DatasetRecipe(TypedDict):
    """Dataset source recorded in a training recipe."""

    repo: str
    path: str
    screen_field: Literal["screen"]
    safety_field: Literal["safety"]


class LabelRecipe(TypedDict):
    """Trainable output labels recorded in a training recipe."""

    screen: list[str]
    safety: list[str]


class PreprocessRecipe(TypedDict):
    """Image preprocessing recorded in a training recipe."""

    resize_longest_side_px: int
    preserve_aspect_ratio: bool
    crop: bool
    stretch: bool
    horizontal_flip: bool
    mean: list[float]
    std: list[float]


class AugmentationRecipe(TypedDict):
    """Training augmentation choices."""

    brightness_probability: float
    brightness_min: float
    brightness_max: float
    contrast_probability: float
    contrast_min: float
    contrast_max: float
    artifact_probability: float
    artifact_quality_min: int
    artifact_quality_max: int
    artifact_codecs: list[str]


class TrainingRunRecipe(TypedDict):
    """Epoch and iterator settings."""

    epochs: int
    batch_size: int
    workers: int
    seed: int
    grad_accum_steps: int
    effective_batch_size: int
    target_effective_batch_size: int


class OptimizerRecipe(TypedDict):
    """Optimizer settings."""

    lr: float
    backbone_lr: float
    head_lr: float
    weight_decay: float


class SamplingRecipe(TypedDict):
    """Adaptive sampling settings."""

    mode: Literal["adaptive_label_aspect_bucketed"]
    min_train_count: int
    screen_target_ratio: float
    safety_target_ratio: float
    screen_max_repeat: int
    safety_max_repeat: int
    folded_screen_labels: dict[str, str]


class TaskScheduleRecipe(TypedDict):
    """Explicit multi-task batch and loss policy."""

    policy: Literal["deterministic_cover_both_loaders"]
    screen_batch_losses: list[Literal["screen", "safety"]]
    safety_batch_losses: list[Literal["safety"]]
    safety_training_batches: Literal[True]


class LossRecipe(TypedDict):
    """Loss functions and effective task weighting."""

    screen: Literal["cross_entropy"]
    safety: Literal["cross_entropy"]
    safety_loss_weight: float
    safety_batch_probability: float
    task_schedule: TaskScheduleRecipe


class CheckpointScoreRecipe(TypedDict):
    """Serialized checkpoint scoring formula."""

    accuracy_weight: float
    formula: str


class EvaluationRecipe(TypedDict):
    """Evaluation and checkpoint selection settings."""

    eval_class_limit: int
    checkpoint_score: CheckpointScoreRecipe


class CountRecipe(TypedDict):
    """Screen and safety label counts by split."""

    train_screen: dict[str, int]
    val_screen: dict[str, int]
    test_screen: dict[str, int]
    train_safety: dict[str, int]
    val_safety: dict[str, int]
    test_safety: dict[str, int]


class TrainingRecipe(TypedDict):
    """Complete versioned training recipe persisted as JSON."""

    schema: Literal[4]
    dataset: DatasetRecipe
    model: str
    labels: LabelRecipe
    preprocess: PreprocessRecipe
    augmentation: AugmentationRecipe
    training: TrainingRunRecipe
    optimizer: OptimizerRecipe
    sampling: SamplingRecipe
    loss: LossRecipe
    evaluation: EvaluationRecipe
    counts: CountRecipe
