"""Fit the training model and evaluate its native checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.state.export import ModelRun
from src.domains.model.network import device
from src.state.loop import EvalRun, TrainRun
from src.state.checkpoint import ModelCheckpoint
from src.domains.onnx import eval_artifact_stages
from src.domains.report.build import previous_val_metrics
from src.domains.artifacts.archive import read_checkpoint_epoch
from src.domains.samples.iterators import build_batch_iterators
from src.domains.artifacts.checkpoints import read_model_checkpoint
from src.domains.evaluation.diagnostics import reset_eval_artifacts
from src.domains.epochs import (
    fit_epochs,
    build_model,
    build_state,
    collect_val_metrics,
    resume_from_checkpoint,
)
from src.config.artifacts import (
    TRAIN_CHECKPOINTS_DIR,
    CURRENT_CHECKPOINT_FILENAME,
    TRAIN_BEST_CHECKPOINT_FILENAME,
)

if TYPE_CHECKING:
    from src.state.metrics import ValidationMetrics
    from src.state.export import PreparedTraining, TrainJob, ValMetrics


def fit_train_model(job: TrainJob, prepared: PreparedTraining) -> ModelRun:
    """Fit or resume the model, then load the selected checkpoint."""
    args = job.args
    dashboard = job.dashboard
    run_device = device()
    dashboard.device_name = run_device.type
    dashboard.refresh(force=True)
    iterators = build_batch_iterators(prepared.datasets, prepared.ids, args, run_device, dashboard)
    model = build_model(args, job.model_id, prepared.labels, run_device, dashboard)
    current_checkpoint = job.run_dir / TRAIN_CHECKPOINTS_DIR / CURRENT_CHECKPOINT_FILENAME
    eval_only = args.resume and read_checkpoint_epoch(current_checkpoint) >= args.epochs
    dashboard.eval_only = eval_only
    dashboard.refresh(force=True)
    if not eval_only:
        state = build_state(model, args, run_device)
        train_run = TrainRun(
            args=args,
            labels=prepared.labels,
            iterators=iterators,
            state=state,
            run_device=run_device,
            run_dir=job.run_dir,
            dashboard=dashboard,
        )
        resume_from_checkpoint(train_run)
        fit_epochs(train_run)
    reset_eval_artifacts(job.run_dir, eval_artifact_stages(args))
    checkpoint_filename = checkpoint_for_export(args.export_checkpoint)
    read_model_checkpoint(
        ModelCheckpoint(
            path=job.run_dir / TRAIN_CHECKPOINTS_DIR / checkpoint_filename,
            model=model,
            run_device=run_device,
            screen_labels=prepared.labels.screen,
            safety_labels=prepared.labels.safety,
        ),
    )
    return ModelRun(iterators=iterators, model=model, run_device=run_device, eval_only=eval_only)


def checkpoint_for_export(selection: str) -> str:
    """Return the checkpoint artifact selected for export."""
    if selection == "latest":
        return CURRENT_CHECKPOINT_FILENAME
    return TRAIN_BEST_CHECKPOINT_FILENAME


def validation_metrics(metrics: ValMetrics) -> ValidationMetrics:
    """Evaluate the selected checkpoint or reuse validated prior metrics."""
    if metrics.eval_only:
        return previous_val_metrics(metrics.run_dir)
    return collect_val_metrics(
        EvalRun(
            model=metrics.model,
            labels=metrics.labels,
            iterators=metrics.iterators,
            run_device=metrics.run_device,
            dashboard=metrics.dashboard,
            run_dir=metrics.run_dir,
        ),
    )
