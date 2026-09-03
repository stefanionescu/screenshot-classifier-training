"""Own epoch scheduling, validation, scoring, and checkpoint updates."""

from __future__ import annotations

import torch
from torch import nn
from typing import TYPE_CHECKING
from src.domains.labels import IGNORED_LABEL_ID
from src.domains.artifacts.files import file_digest
from src.domains.evaluation.metrics import evaluate
from src.state.eval import EvalArtifacts, TorchEval
from src.domains.evaluation.stats import checkpoint_score
from src.state.checkpoint import Checkpoint, SavedCheckpoint
from src.state.metrics import EvaluationMetrics, ValidationMetrics
from src.state.training import BatchIterators, LabelState, ModelState
from src.domains.artifacts.runs import append_jsonl, metrics_dir, train_config_path
from src.domains.lightning import fit_lightning_epoch, grad_accum_steps, setup_lightning_run
from src.domains.model.network import MultiTaskClassifier, build_optimizer, check_backbone_features
from src.config.artifacts import (
    TRAIN_CHECKPOINTS_DIR,
    TRAIN_EPOCHS_FILENAME,
    CURRENT_CHECKPOINT_FILENAME,
    TRAIN_BEST_CHECKPOINT_FILENAME,
)
from src.domains.artifacts.checkpoints import (
    read_checkpoint,
    save_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path
    from src.state.types import TrainModel
    from src.state.training import TrainArgs
    from src.state.loop import EvalRun, TrainRun
    from src.state.dashboard import TrainDashboardProtocol

STAGE_CHECKPOINT = "checkpoint"
STAGE_RESUME = "resume"
STAGE_MODEL = "model"
VAL_FULL = "val full"
VAL_SAFETY_BALANCED = "val safety balanced"
VAL_SCREEN_BALANCED = "val screen balanced"


def build_model(
    args: TrainArgs,
    model_id: str,
    labels: LabelState,
    run_device: torch.device,
    dashboard: TrainDashboardProtocol,
) -> MultiTaskClassifier:
    """Build model."""
    dashboard.set_stage(STAGE_MODEL, 0, 2)
    model = MultiTaskClassifier(
        model_id,
        len(labels.screen),
        len(labels.safety),
    ).to(run_device)
    dashboard.advance_stage()
    check_backbone_features(model, args.image_size, run_device)
    dashboard.advance_stage()
    return model


def build_state(
    model: MultiTaskClassifier,
    args: TrainArgs,
    run_device: torch.device,
) -> ModelState:
    """Build state."""
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return ModelState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=torch.GradScaler("cuda", enabled=run_device.type == "cuda"),
        screen_criterion=nn.CrossEntropyLoss(),
        safety_criterion=nn.CrossEntropyLoss(ignore_index=IGNORED_LABEL_ID, reduction="sum"),
        start_epoch=0,
        best_score=-1.0,
        best_epoch=0,
    )


def resume_from_checkpoint(run: TrainRun) -> None:
    """Restore a compatible latest checkpoint for a resumed run."""
    if not run.args.resume:
        return
    run.dashboard.set_stage(STAGE_RESUME, 0, 1)
    run.state.start_epoch, run.state.best_score, run.state.best_epoch = read_checkpoint(
        SavedCheckpoint(
            path=run.run_dir / TRAIN_CHECKPOINTS_DIR / CURRENT_CHECKPOINT_FILENAME,
            model=run.state.model,
            optimizer=run.state.optimizer,
            scheduler=run.state.scheduler,
            scaler=run.state.scaler,
            run_device=run.run_device,
            screen_labels=run.labels.screen,
            safety_labels=run.labels.safety,
            screen_sampler=run.iterators.screen_sampler,
            safety_sampler=run.iterators.safety_sampler,
            model_id=run.args.model,
            image_size=run.args.image_size,
            train_config_sha256=file_digest(train_config_path(run.run_dir))["sha256"],
        ),
    )
    run.dashboard.advance_stage()


def fit_epochs(run: TrainRun) -> None:
    """Fit epochs."""
    fabric = setup_lightning_run(run)
    accumulation_steps = grad_accum_steps(
        run.args.micro_batch_size,
        run.args.grad_accum_steps,
        run.args.batch_size,
    )

    for epoch in range(run.state.start_epoch + 1, run.args.epochs + 1):
        run.iterators.screen_sampler.epoch = epoch
        run.iterators.safety_sampler.epoch = epoch
        train_losses = fit_lightning_epoch(run, fabric, epoch, accumulation_steps)
        score = score_epoch(run, epoch)
        append_epoch_history(run.run_dir, epoch, train_losses, score, run.state)
        save_current_checkpoint(run, epoch)
        save_best_checkpoint(run, epoch)


def score_epoch(run: TrainRun, epoch: int) -> float:
    """Evaluate one epoch and update its best-checkpoint score."""
    state = run.state
    val_screen_metrics, val_safety_metrics = collect_validation(
        state.model,
        run.labels,
        run.iterators,
        run.run_device,
        run.dashboard,
    )
    score = checkpoint_score(val_screen_metrics, val_safety_metrics)
    record_best(state, score, epoch)
    run.dashboard.set_validation(val_screen_metrics, val_safety_metrics, score, state.best_score, state.best_epoch)
    return score


def append_epoch_history(
    run_dir: Path,
    epoch: int,
    train_losses: dict[str, float],
    score: float,
    state: ModelState,
) -> None:
    """Append epoch history."""
    append_jsonl(
        metrics_dir(run_dir) / TRAIN_EPOCHS_FILENAME,
        {
            "epoch": epoch,
            "lr": float(state.optimizer.param_groups[-1]["lr"]),
            **train_losses,
            "val_score": score,
            "best_score": state.best_score,
            "best_epoch": state.best_epoch,
        },
    )


def record_best(state: ModelState, score: float, epoch: int) -> None:
    """Record best."""
    if score <= state.best_score:
        return
    state.best_score = score
    state.best_epoch = epoch


def collect_validation(
    model: TrainModel,
    labels: LabelState,
    iterators: BatchIterators,
    run_device: torch.device,
    dashboard: TrainDashboardProtocol,
) -> tuple[EvaluationMetrics, EvaluationMetrics]:
    """Collect validation."""
    metrics = evaluate(
        TorchEval(
            model=model,
            iterator=iterators.val,
            run_device=run_device,
            labels=labels,
            description=VAL_FULL,
            dashboard=dashboard,
        ),
    )
    return metrics, metrics


def save_current_checkpoint(run: TrainRun, epoch: int) -> None:
    """Save latest checkpoint."""
    run.dashboard.set_stage(STAGE_CHECKPOINT, 0, 2)
    save_checkpoint(
        Checkpoint(
            path=run.run_dir / TRAIN_CHECKPOINTS_DIR / CURRENT_CHECKPOINT_FILENAME,
            model=run.state.model,
            optimizer=run.state.optimizer,
            scheduler=run.state.scheduler,
            scaler=run.state.scaler,
            epoch=epoch,
            best_score=run.state.best_score,
            best_epoch=run.state.best_epoch,
            screen_labels=run.labels.screen,
            safety_labels=run.labels.safety,
            screen_sampler=run.iterators.screen_sampler,
            safety_sampler=run.iterators.safety_sampler,
            args=run.args,
            run_dir=run.run_dir,
        ),
    )
    run.dashboard.advance_stage()


def save_best_checkpoint(run: TrainRun, epoch: int) -> None:
    """Save best checkpoint."""
    if run.state.best_epoch == epoch:
        save_checkpoint(
            Checkpoint(
                path=run.run_dir / TRAIN_CHECKPOINTS_DIR / TRAIN_BEST_CHECKPOINT_FILENAME,
                model=run.state.model,
                optimizer=run.state.optimizer,
                scheduler=run.state.scheduler,
                scaler=run.state.scaler,
                epoch=epoch,
                best_score=run.state.best_score,
                best_epoch=run.state.best_epoch,
                screen_labels=run.labels.screen,
                safety_labels=run.labels.safety,
                screen_sampler=run.iterators.screen_sampler,
                safety_sampler=run.iterators.safety_sampler,
                args=run.args,
                run_dir=run.run_dir,
            ),
        )
    run.dashboard.advance_stage()


def collect_val_metrics(run: EvalRun) -> ValidationMetrics:
    """Collect val metrics."""
    return ValidationMetrics(
        full=evaluate(
            TorchEval(
                model=run.model,
                iterator=run.iterators.val,
                run_device=run.run_device,
                labels=run.labels,
                description=VAL_FULL,
                dashboard=run.dashboard,
                artifacts=EvalArtifacts(run.run_dir, "val_full", "val") if run.run_dir is not None else None,
            ),
        ),
        screen_balanced=evaluate(
            TorchEval(
                model=run.model,
                iterator=run.iterators.val_screen,
                run_device=run.run_device,
                labels=run.labels,
                description=VAL_SCREEN_BALANCED,
                dashboard=run.dashboard,
                artifacts=EvalArtifacts(run.run_dir, "val_screen_balanced", "val") if run.run_dir is not None else None,
            ),
        ),
        safety_balanced=evaluate(
            TorchEval(
                model=run.model,
                iterator=run.iterators.val_safety,
                run_device=run.run_device,
                labels=run.labels,
                description=VAL_SAFETY_BALANCED,
                dashboard=run.dashboard,
                artifacts=EvalArtifacts(run.run_dir, "val_safety_balanced", "val") if run.run_dir is not None else None,
            ),
        ),
    )
