"""Coordinate the independently owned training phases."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.state.train.export import TrainJob
from src.domains.train.fit import fit_train_model
from src.shared.console import write_command_fields
from src.shared.runtime.progress import ProgressBar
from src.domains.train.export import export_training
from src.domains.train.dashboard import TrainDashboard
from src.domains.train.prepare import prepare_training
from src.config.train.artifacts import TRAIN_EXPORT_DIR
from src.domains.train.artifacts.export import export_ready
from src.domains.train.report.build import write_training_report
from src.domains.train.resume import read_train_config, resume_args
from src.domains.train.artifacts.paths import repo_root, safe_name, target_repo_id
from src.domains.train.hub import normalize_model_id, publish_training_export, push_export, require_hf_token

if TYPE_CHECKING:
    from src.state.train.training import TrainArgs


def run_train(args: TrainArgs) -> None:
    """Resolve the training job and execute the requested workflow."""
    token = require_hf_token() if args.push or args.push_only else None
    model_id = normalize_model_id(args.model)
    run_dir = (
        repo_root()
        / "output"
        / "models"
        / safe_name(args.output_dir, "--output")
        / safe_name(model_id.replace("/", "__").replace(":", "__"), "--model")
    )
    export_dir = run_dir / TRAIN_EXPORT_DIR
    repo_id = target_repo_id(args.repo, args.output_dir)
    if args.push_only:
        if not export_ready(run_dir, export_dir, args):
            msg = "export is incomplete. Run train without --push-only before pushing."
            raise TrainingError(msg)
        if token is None:
            msg = "Hugging Face token is required for push-only mode."
            raise TrainingError(msg)
        write_command_fields(
            None,
            (
                ("task", "Model publish"),
                ("source", str(export_dir)),
                ("repository", repo_id),
                ("visibility", "public" if args.public else "private"),
            ),
        )
        with ProgressBar(1) as progress:
            push_export(export_dir, repo_id, token, is_private=not args.public)
            progress.update(1)
        write_command_fields("Model published", (("repository", repo_id), ("status", "complete")))
        return

    saved_config = read_train_config(run_dir) if args.resume else None
    args = resume_args(args, saved_config)
    with TrainDashboard(args, model_id, repo_id, eval_only=args.resume) as dashboard:
        run_training(
            TrainJob(
                args=args,
                token=token,
                model_id=model_id,
                run_dir=run_dir,
                export_dir=export_dir,
                repo_id=repo_id,
                saved_config=saved_config,
                dashboard=dashboard,
            ),
        )


def run_training(job: TrainJob) -> None:
    """Run each training phase in lifecycle order."""
    prepared = prepare_training(job)
    run = fit_train_model(job, prepared)
    evaluation_summary = export_training(job, prepared, run)
    write_training_report(job, prepared, run, evaluation_summary)
    publish_training_export(job.args, job.token, job.repo_id, job.export_dir, job.dashboard)
