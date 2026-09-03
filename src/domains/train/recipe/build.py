"""Build the typed training recipe JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.domains.train.artifacts.paths import repo_root
from src.config.train.metrics import CHECKPOINT_ACCURACY_WEIGHT
from src.config.train.model import (
    DATASET_REPO,
    AUG_CONTRAST_MAX,
    AUG_CONTRAST_MIN,
    AUG_BRIGHTNESS_MAX,
    AUG_BRIGHTNESS_MIN,
    ARTIFACT_AUG_CODECS,
    BACKBONE_LR_DIVISOR,
    ARTIFACT_AUG_PROBABILITY,
    ARTIFACT_AUG_QUALITY_MAX,
    ARTIFACT_AUG_QUALITY_MIN,
    AUG_CONTRAST_PROBABILITY,
    AUG_BRIGHTNESS_PROBABILITY,
)
from src.state.train.recipe import (
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
    from pathlib import Path
    from src.state.train.inputs import TrainConfig

CHECKPOINT_SCORE_FORMULA = (
    "accuracy_weight * screen_accuracy + screen_macro_f1 + fallback_recall + other_recall + "
    "accuracy_weight * safety_accuracy + safety_macro_f1 + safety_balanced_accuracy"
)


def train_config(config: TrainConfig) -> TrainingRecipe:
    """Build the complete versioned training recipe."""
    return TrainingRecipe(
        schema=4,
        dataset=dataset_recipe(config),
        model=config.model_id,
        labels=LabelRecipe(screen=config.labels.screen, safety=config.labels.safety),
        preprocess=preprocess_recipe(config),
        augmentation=augmentation_recipe(),
        training=training_recipe(config),
        optimizer=optimizer_recipe(config),
        sampling=sampling_recipe(config),
        loss=loss_recipe(config),
        evaluation=evaluation_recipe(config),
        counts=count_config(config),
    )


def dataset_recipe(config: TrainConfig) -> DatasetRecipe:
    """Build the dataset source section."""
    return DatasetRecipe(
        repo=DATASET_REPO,
        path=train_path(config.dataset_path),
        screen_field="screen",
        safety_field="safety",
    )


def preprocess_recipe(config: TrainConfig) -> PreprocessRecipe:
    """Build the image preprocessing section."""
    return PreprocessRecipe(
        resize_longest_side_px=config.args.image_size,
        preserve_aspect_ratio=True,
        crop=False,
        stretch=False,
        horizontal_flip=False,
        mean=list(config.preprocess.mean),
        std=list(config.preprocess.std),
    )


def augmentation_recipe() -> AugmentationRecipe:
    """Build the fixed augmentation section."""
    return AugmentationRecipe(
        brightness_probability=AUG_BRIGHTNESS_PROBABILITY,
        brightness_min=AUG_BRIGHTNESS_MIN,
        brightness_max=AUG_BRIGHTNESS_MAX,
        contrast_probability=AUG_CONTRAST_PROBABILITY,
        contrast_min=AUG_CONTRAST_MIN,
        contrast_max=AUG_CONTRAST_MAX,
        artifact_probability=ARTIFACT_AUG_PROBABILITY,
        artifact_quality_min=ARTIFACT_AUG_QUALITY_MIN,
        artifact_quality_max=ARTIFACT_AUG_QUALITY_MAX,
        artifact_codecs=[str(codec) for codec in ARTIFACT_AUG_CODECS],
    )


def training_recipe(config: TrainConfig) -> TrainingRunRecipe:
    """Build epoch and iterator settings."""
    args = config.args
    return TrainingRunRecipe(
        epochs=args.epochs,
        batch_size=args.micro_batch_size,
        workers=args.workers,
        seed=args.seed,
        grad_accum_steps=config.accumulation_steps,
        effective_batch_size=args.micro_batch_size * config.accumulation_steps,
        target_effective_batch_size=args.batch_size,
    )


def optimizer_recipe(config: TrainConfig) -> OptimizerRecipe:
    """Build optimizer settings."""
    args = config.args
    return OptimizerRecipe(
        lr=args.lr,
        backbone_lr=args.lr / BACKBONE_LR_DIVISOR,
        head_lr=args.lr,
        weight_decay=args.weight_decay,
    )


def sampling_recipe(config: TrainConfig) -> SamplingRecipe:
    """Build adaptive sampling settings."""
    args = config.args
    return SamplingRecipe(
        mode="adaptive_label_aspect_bucketed",
        min_train_count=args.min_train_count,
        screen_target_ratio=args.screen_target_ratio,
        safety_target_ratio=args.safety_target_ratio,
        screen_max_repeat=args.screen_max_repeat,
        safety_max_repeat=args.safety_max_repeat,
        folded_screen_labels=config.fold_map,
    )


def loss_recipe(config: TrainConfig) -> LossRecipe:
    """Build the explicit batch-to-loss weighting policy."""
    return LossRecipe(
        screen="cross_entropy",
        safety="cross_entropy",
        safety_loss_weight=config.args.safety_loss_weight,
        safety_batch_probability=config.args.safety_batch_probability,
        task_schedule=TaskScheduleRecipe(
            policy="deterministic_cover_both_loaders",
            screen_batch_losses=["screen", "safety"],
            safety_batch_losses=["safety"],
            safety_training_batches=True,
        ),
    )


def evaluation_recipe(config: TrainConfig) -> EvaluationRecipe:
    """Build evaluation and checkpoint-scoring settings."""
    return EvaluationRecipe(
        eval_class_limit=config.args.eval_class_limit,
        checkpoint_score=CheckpointScoreRecipe(
            accuracy_weight=CHECKPOINT_ACCURACY_WEIGHT,
            formula=CHECKPOINT_SCORE_FORMULA,
        ),
    )


def count_config(config: TrainConfig) -> CountRecipe:
    """Build screen and safety counts by split."""
    return CountRecipe(
        train_screen=config.counts.train,
        val_screen=config.counts.val,
        test_screen=config.counts.test,
        train_safety=config.safety_counts.train,
        val_safety=config.safety_counts.val,
        test_safety=config.safety_counts.test,
    )


def train_path(path: Path) -> str:
    """Return a repository-relative training path."""
    resolved = path if path.is_absolute() else repo_root() / path
    return str(resolved.resolve().relative_to(repo_root()))
