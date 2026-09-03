"""Build and persist typed training reports."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.state.export import ReportInputs
from src.domains.artifacts.paths import repo_root
from src.domains.model.report import write_report_markdown
from src.state.metrics import SamplerReport, TrainingReport
from src.domains.artifacts.files import read_json, write_json
from src.domains.report.schema import parse_validation_metrics
from src.domains.artifacts.runs import metrics_dir, report_json_path
from src.domains.images.skipped.summary import skipped_images_summary
from src.config.artifacts import (
    TRAIN_CHECKPOINTS_DIR,
    TRAIN_BEST_CHECKPOINT_FILENAME,
    TRAIN_REPORT_MARKDOWN_FILENAME,
)

if TYPE_CHECKING:
    from pathlib import Path
    from src.state.dashboard import TrainDashboardProtocol
    from src.state.training import BatchIterators, LabelState
    from src.state.export import ModelRun, PreparedTraining, TrainJob
    from src.state.metrics import EvaluationSummary, ValidationMetrics

REPORT_JSON_STAGE = "report json"
REPORT_PAGE_STAGE = "report page"


def write_training_report(
    job: TrainJob,
    prepared: PreparedTraining,
    run: ModelRun,
    evaluation_summary: EvaluationSummary,
) -> None:
    """Build and persist the machine-readable and Markdown reports."""
    report = build_report(
        ReportInputs(
            args=job.args,
            model_id=job.model_id,
            run_dir=job.run_dir,
            profile=prepared.profile,
            labels=prepared.labels,
            iterators=run.iterators,
            evaluation_summary=evaluation_summary,
        ),
    )
    write_report_json(job.run_dir, report, job.dashboard)
    write_report_page(job.run_dir, report, job.dashboard)


def build_report(report: ReportInputs) -> TrainingReport:
    """Build the final serialized training report."""
    metrics = report.evaluation_summary
    return TrainingReport(
        model=report.model_id,
        image_size=report.args.image_size,
        screen_labels=report.labels.screen,
        safety_labels=report.labels.safety,
        best_checkpoint=str(
            (report.run_dir / TRAIN_CHECKPOINTS_DIR / TRAIN_BEST_CHECKPOINT_FILENAME).relative_to(repo_root()),
        ),
        profile=report.profile,
        skipped_images=skipped_images_summary(report.run_dir),
        sampling=sampler_report(report.iterators, report.labels),
        val_full=metrics.validation.full,
        val_screen_balanced=metrics.validation.screen_balanced,
        val_safety_balanced=metrics.validation.safety_balanced,
        test_full=metrics.test["test_full"],
        test_screen_balanced=metrics.test["test_screen_balanced"],
        test_safety_balanced=metrics.test["test_safety_balanced"],
        onnx_models=metrics.onnx_models,
    )


def sampler_report(iterators: BatchIterators, labels: LabelState) -> SamplerReport:
    """Build sampler coverage and epoch-size metadata."""
    return SamplerReport(
        mode="adaptive_label_aspect_bucketed",
        screen_train_batches=len(iterators.train_screen),
        safety_train_batches=len(iterators.train_safety),
        screen_labels=labels.screen,
        safety_labels=labels.safety,
        screen_coverage=iterators.screen_sampler.coverage_stats(),
        safety_coverage=iterators.safety_sampler.coverage_stats(),
    )


def write_report_json(run_dir: Path, report: TrainingReport, dashboard: TrainDashboardProtocol) -> None:
    """Write report JSON."""
    dashboard.set_stage(REPORT_JSON_STAGE, 0, 1)
    write_json(report_json_path(run_dir), report)
    dashboard.advance_stage()


def write_report_page(run_dir: Path, report: TrainingReport, dashboard: TrainDashboardProtocol) -> None:
    """Write report Markdown."""
    dashboard.set_stage(REPORT_PAGE_STAGE, 0, 1)
    write_report_markdown(metrics_dir(run_dir) / TRAIN_REPORT_MARKDOWN_FILENAME, report)
    dashboard.advance_stage()


def previous_val_metrics(run_dir: Path) -> ValidationMetrics:
    """Return validated reusable metrics from a previous report."""
    path = report_json_path(run_dir)
    if not path.is_file():
        msg = "evaluation-only resume requires an existing training report."
        raise TrainingError(msg)
    return parse_validation_metrics(read_json(path))
