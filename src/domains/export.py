"""Stage, validate, and publish local training export artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.domains.fit import validation_metrics
from src.state.metrics import EvaluationSummary
from src.domains.model.card import write_model_card
from src.domains.model.export import staged_model_export
from src.domains.evaluation.diagnostics import write_prediction_analysis
from src.domains.onnx import BASE_EVAL_ARTIFACT_STAGES, collect_onnx_metrics
from src.state.export import ModelCard, ModelExport, OnnxMetrics, ValMetrics
from src.domains.artifacts.export import copy_checkpoint, copy_train_config, export_ready

if TYPE_CHECKING:
    from src.state.export import ModelRun, PreparedTraining, TrainJob

EXPORT_STAGE = "export"


def export_training(job: TrainJob, prepared: PreparedTraining, run: ModelRun) -> EvaluationSummary:
    """Create a complete export in staging and atomically publish it locally."""
    args = job.args
    validation = validation_metrics(
        ValMetrics(
            eval_only=run.eval_only,
            run_dir=job.run_dir,
            model=run.model,
            labels=prepared.labels,
            iterators=run.iterators,
            run_device=run.run_device,
            dashboard=job.dashboard,
        ),
    )
    needs_export = not run.eval_only or not export_ready(job.run_dir, job.export_dir, args)
    export = ModelExport(
        model=run.model,
        export_dir=job.export_dir,
        model_id=job.model_id,
        labels=prepared.labels,
        image_size=args.image_size,
        args=args,
    )
    job.dashboard.set_stage(EXPORT_STAGE, 0, 1)
    with staged_model_export(export, is_existing_copied=not needs_export) as staging_dir:
        copy_checkpoint(job.run_dir, staging_dir, args)
        copy_train_config(job.run_dir, staging_dir)
        onnx_models, test_metrics = collect_onnx_metrics(
            OnnxMetrics(
                export_dir=staging_dir,
                run_dir=job.run_dir,
                labels=prepared.labels,
                iterators=run.iterators,
                args=args,
                dashboard=job.dashboard,
            ),
        )
        evaluation_summary = EvaluationSummary(validation=validation, test=test_metrics, onnx_models=onnx_models)
        write_model_card(
            ModelCard(
                export_dir=staging_dir,
                model_id=job.model_id,
                repo_id=job.repo_id,
                report=evaluation_summary,
            ),
        )
        if not export_ready(job.run_dir, staging_dir, args):
            msg = "staged export failed final integrity validation."
            raise TrainingError(msg)
    job.dashboard.advance_stage()
    write_prediction_analysis(job.run_dir, BASE_EVAL_ARTIFACT_STAGES)
    job.dashboard.evaluation_summary = evaluation_summary
    job.dashboard.refresh(force=True)
    return evaluation_summary
