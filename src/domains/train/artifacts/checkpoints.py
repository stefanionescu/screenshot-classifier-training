"""Persist and restore versioned training checkpoint state."""

from __future__ import annotations

import math
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
from src.domains.train.artifacts.files import file_digest
from src.domains.train.artifacts.run_files import train_config_path
from src.domains.train.artifacts.randomness import random_state, restore_random_state, validate_random_state
from src.domains.train.artifacts.archive import (
    CheckpointMap,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_ARCHITECTURE,
    read_checkpoint_archive,
    write_checkpoint_archive,
)

if TYPE_CHECKING:
    from src.state.train.checkpoint import Checkpoint, ModelCheckpoint, SavedCheckpoint


def save_checkpoint(checkpoint: Checkpoint) -> None:
    """Persist all model, optimizer, sampler, identity, and RNG state."""
    payload: CheckpointMap = {
        "schema": CHECKPOINT_SCHEMA,
        "model": checkpoint.model.state_dict(),
        "optimizer": checkpoint.optimizer.state_dict(),
        "scheduler": checkpoint.scheduler.state_dict(),
        "scaler": checkpoint.scaler.state_dict(),
        "epoch": checkpoint.epoch,
        "best_score": checkpoint.best_score,
        "best_epoch": checkpoint.best_epoch,
        "screen_labels": checkpoint.screen_labels,
        "safety_labels": checkpoint.safety_labels,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "image_size": checkpoint.args.image_size,
        "model_id": checkpoint.args.model,
        "samplers": {
            "screen": checkpoint.screen_sampler.state_dict(),
            "safety": checkpoint.safety_sampler.state_dict(),
        },
        "train_config_sha256": file_digest(train_config_path(checkpoint.run_dir))["sha256"],
        "rng": random_state(),
    }
    write_checkpoint_archive(checkpoint.path, payload)


def read_checkpoint(checkpoint_file: SavedCheckpoint) -> tuple[int, float, int]:
    """Validate a checkpoint completely before restoring training state."""
    checkpoint = read_checkpoint_archive(checkpoint_file.path, checkpoint_file.run_device)
    validate_checkpoint_identity(checkpoint, checkpoint_file)
    screen_state, safety_state = checkpoint_sampler_states(checkpoint)
    checkpoint_file.screen_sampler.validated_snapshot(screen_state)
    checkpoint_file.safety_sampler.validated_snapshot(safety_state)
    model_state, optimizer_state, scheduler_state = training_maps(checkpoint)
    scaler_state = checkpoint.get("scaler")
    if not isinstance(scaler_state, dict):
        msg = "checkpoint has invalid gradient-scaler state."
        raise TrainingError(msg)
    rng_state = validate_random_state(checkpoint.get("rng"))
    scores = checkpoint_scores(checkpoint)

    checkpoint_file.model.load_state_dict(model_state)
    checkpoint_file.optimizer.load_state_dict(optimizer_state)
    checkpoint_file.scheduler.load_state_dict(scheduler_state)
    if scaler_state:
        checkpoint_file.scaler.load_state_dict(cast("CheckpointMap", scaler_state))
    checkpoint_file.screen_sampler.restore_state(screen_state)
    checkpoint_file.safety_sampler.restore_state(safety_state)
    restore_random_state(rng_state)
    return scores


def validate_checkpoint_identity(checkpoint: CheckpointMap, checkpoint_file: SavedCheckpoint) -> None:
    """Validate checkpoint architecture and immutable run identity fields."""
    checks = (
        (
            checkpoint.get("architecture") != CHECKPOINT_ARCHITECTURE,
            "Checkpoint architecture does not match flat training. Start a fresh run.",
        ),
        (
            checkpoint.get("screen_labels") != checkpoint_file.screen_labels,
            "Checkpoint screen labels do not match selected labels.",
        ),
        (
            checkpoint.get("safety_labels") != checkpoint_file.safety_labels,
            "Checkpoint safety labels do not match enabled safety labels.",
        ),
        (
            checkpoint.get("model_id") != checkpoint_file.model_id,
            "Checkpoint model ID does not match the current run.",
        ),
        (
            checkpoint.get("image_size") != checkpoint_file.image_size,
            "Checkpoint image size does not match the current run.",
        ),
        (
            checkpoint.get("train_config_sha256") != checkpoint_file.train_config_sha256,
            "Checkpoint training configuration does not match the current run.",
        ),
    )
    for invalid, message in checks:
        if invalid:
            raise TrainingError(message)


def training_maps(checkpoint: CheckpointMap) -> tuple[CheckpointMap, CheckpointMap, CheckpointMap]:
    """Return validated model, optimizer, and scheduler states."""
    model_state = checkpoint.get("model")
    optimizer_state = checkpoint.get("optimizer")
    scheduler_state = checkpoint.get("scheduler")
    if (
        not isinstance(model_state, dict)
        or not isinstance(optimizer_state, dict)
        or not isinstance(scheduler_state, dict)
    ):
        msg = "checkpoint is missing model training state."
        raise TrainingError(msg)
    return (
        cast("CheckpointMap", model_state),
        cast("CheckpointMap", optimizer_state),
        cast("CheckpointMap", scheduler_state),
    )


def checkpoint_scores(checkpoint: CheckpointMap) -> tuple[int, float, int]:
    """Return validated checkpoint score fields."""
    epoch_value = checkpoint.get("epoch")
    best_score = checkpoint.get("best_score")
    best_epoch = checkpoint.get("best_epoch")
    if (
        isinstance(epoch_value, bool)
        or not isinstance(epoch_value, int)
        or epoch_value < 0
        or isinstance(best_score, bool)
        or not isinstance(best_score, int | float)
        or not math.isfinite(float(best_score))
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 0 <= best_epoch <= epoch_value
    ):
        msg = "checkpoint has invalid training metrics."
        raise TrainingError(msg)
    return epoch_value, float(best_score), best_epoch


def read_model_checkpoint(model_file: ModelCheckpoint) -> None:
    """Restore model weights from a compatible validated checkpoint."""
    checkpoint = read_checkpoint_archive(model_file.path, model_file.run_device)
    checks = (
        (
            checkpoint.get("architecture") != CHECKPOINT_ARCHITECTURE,
            "Checkpoint architecture does not match flat training.",
        ),
        (
            checkpoint.get("screen_labels") != model_file.screen_labels,
            "Checkpoint screen labels do not match selected labels.",
        ),
        (
            checkpoint.get("safety_labels") != model_file.safety_labels,
            "Checkpoint safety labels do not match enabled safety labels.",
        ),
    )
    for invalid, message in checks:
        if invalid:
            raise TrainingError(message)
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        msg = "checkpoint is missing model state."
        raise TrainingError(msg)
    model_file.model.load_state_dict(cast("CheckpointMap", model_state))


def checkpoint_sampler_states(checkpoint: CheckpointMap) -> tuple[CheckpointMap, CheckpointMap]:
    """Return validated sampler checkpoint state."""
    sampler_state = checkpoint.get("samplers")
    if not isinstance(sampler_state, dict):
        msg = "checkpoint is missing adaptive sampler state."
        raise TrainingError(msg)
    samplers = cast("CheckpointMap", sampler_state)
    if set(samplers) != {"screen", "safety"}:
        msg = "checkpoint sampler fields do not match the supported schema."
        raise TrainingError(msg)
    screen_state = samplers.get("screen")
    safety_state = samplers.get("safety")
    if not isinstance(screen_state, dict) or not isinstance(safety_state, dict):
        msg = "checkpoint has invalid adaptive sampler state."
        raise TrainingError(msg)
    return cast("CheckpointMap", screen_state), cast("CheckpointMap", safety_state)
