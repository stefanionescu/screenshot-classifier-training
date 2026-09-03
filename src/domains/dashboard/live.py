"""Display live progress and final metrics for a training run."""

from __future__ import annotations

import time
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from typing import TYPE_CHECKING, Self
from rich.console import Console, Group
from src.shared.runtime.format import format_elapsed_time
from src.config.metrics import DASHBOARD_MIN_ELAPSED_SECONDS, DASHBOARD_REFRESH_SECONDS
from src.domains.dashboard.view import (
    format_sparkline,
    build_timing_table,
    format_metric_value,
    format_progress_bar,
    build_evaluation_rows,
    collect_class_recalls,
    collect_timing_values,
)

if TYPE_CHECKING:
    from src.state.training import TrainArgs
    from src.state.dashboard import TrainUpdate
    from src.state.metrics import EvaluationMetrics, EvaluationSummary


class TrainDashboard:
    """Represent train dashboard."""

    best_epoch: int | None
    best_score: float | None
    evaluation_summary: EvaluationSummary | None
    learning_rate: float | None
    safety_balanced_accuracy: float | None
    safety_macro_f1: float | None
    safety_top2_accuracy: float | None
    score: float | None
    screen_balanced_accuracy: float | None
    screen_macro_f1: float | None
    screen_top2_accuracy: float | None
    safety_loss: float | None
    screen_loss: float | None
    train_loss: float | None

    def __init__(self, args: TrainArgs, model_id: str, repo_id: str, *, eval_only: bool = False) -> None:
        """Create a dashboard for one training run."""
        self.args, self.model_id, self.repo_id = args, model_id, repo_id
        self.device_name = "pending"
        self.stage = "prepare"
        self.batch = self.batches = self.epoch = 0
        self.eval_done = self.eval_total = 0
        self.processed_images = self.safety_count = self.screen_count = 0
        self.stage_done = self.test_count = self.train_count = self.val_count = 0
        self.stage_total = 1
        self.images_per_second = 0.0
        self.train_loss = self.screen_loss = self.safety_loss = self.learning_rate = None
        self.score = self.best_score = self.best_epoch = None
        self.screen_macro_f1 = self.screen_balanced_accuracy = self.screen_top2_accuracy = None
        self.safety_macro_f1 = self.safety_balanced_accuracy = self.safety_top2_accuracy = None
        self.safety_recalls: dict[str, float] = {}
        self.evaluation_summary = None
        self.eval_only = eval_only
        self.loss_points: list[float] = []
        self.console = Console()
        self.live: Live | None = None
        self.phase_started_at = time.perf_counter()
        self.last_refresh_at = 0.0

    def __enter__(self) -> Self:
        """Enter the context manager."""
        self.live = Live(
            self.view(),
            console=self.console,
            screen=False,
            auto_refresh=False,
            transient=True,
        )
        self.live.start()
        return self

    def __exit__(self, error_type: object, _error: object, _traceback: object) -> None:
        """Exit the context manager."""
        if self.live is not None:
            self.refresh(force=True)
            self.live.stop()
            self.live = None
        if error_type is None:
            self.console.print(self.view())

    def set_stage(self, stage: str, done: int = 0, total: int = 1) -> None:
        """Set stage."""
        self.stage = stage
        self.stage_done = done
        self.stage_total = max(total, 1)
        self.eval_done = 0
        self.eval_total = 0
        self.phase_started_at = time.perf_counter()
        self.refresh(force=True)

    def advance_stage(self) -> None:
        """Mark one unit of the current preparation stage complete."""
        self.stage_done = min(self.stage_done + 1, self.stage_total)
        self.refresh(force=True)

    def set_counts(
        self,
        screen_count: int,
        safety_count: int,
        train_count: int,
        val_count: int,
        test_count: int,
    ) -> None:
        """Set counts."""
        self.screen_count = screen_count
        self.safety_count = safety_count
        self.train_count = train_count
        self.val_count = val_count
        self.test_count = test_count
        self.refresh(force=True)

    def start_epoch(self, epoch: int, batches: int, learning_rate: float) -> None:
        """Start epoch."""
        self.stage = "train"
        self.epoch = epoch
        self.batch = 0
        self.batches = max(batches, 1)
        self.processed_images = 0
        self.images_per_second = 0.0
        self.learning_rate = learning_rate
        self.phase_started_at = time.perf_counter()
        self.refresh(force=True)

    def update_train(self, update: TrainUpdate) -> None:
        """Update train."""
        self.batch = update.batch
        self.processed_images += update.batch_images
        self.train_loss = update.train_loss
        self.screen_loss = update.screen_loss
        self.safety_loss = update.safety_loss
        self.learning_rate = update.learning_rate
        self.loss_points.append(update.train_loss)
        self.loss_points = self.loss_points[-80:]
        elapsed = max(time.perf_counter() - self.phase_started_at, DASHBOARD_MIN_ELAPSED_SECONDS)
        self.images_per_second = self.processed_images / elapsed
        self.refresh()

    def start_eval(self, stage: str, total: int) -> None:
        """Start eval."""
        self.stage = stage
        self.eval_done = 0
        self.eval_total = max(total, 1)
        self.phase_started_at = time.perf_counter()
        self.refresh(force=True)

    def set_validation(
        self,
        screen_metrics: EvaluationMetrics,
        safety_metrics: EvaluationMetrics,
        score: float,
        best_score: float,
        best_epoch: int,
    ) -> None:
        """Set validation."""
        screen = screen_metrics["screen"]
        safety = safety_metrics["safety"]
        safety_classes = safety["per_class"]
        self.score = score
        self.best_score = best_score
        self.best_epoch = best_epoch
        self.screen_macro_f1 = screen["macro_f1"]
        self.screen_balanced_accuracy = screen["balanced_accuracy"]
        self.screen_top2_accuracy = screen["top2_accuracy"]
        self.safety_macro_f1 = safety["macro_f1"]
        self.safety_balanced_accuracy = safety["balanced_accuracy"]
        self.safety_top2_accuracy = safety["top2_accuracy"]
        self.safety_recalls = collect_class_recalls(safety_classes)
        self.refresh(force=True)

    def refresh(self, *, force: bool = False) -> None:
        """Refresh the live display when its rate limit permits."""
        if self.live is None:
            return
        now = time.perf_counter()
        if force or now - self.last_refresh_at >= DASHBOARD_REFRESH_SECONDS:
            self.live.update(self.view(), refresh=True)
            self.last_refresh_at = now

    def view(self) -> Group:
        """Compose the panels visible for the current run phase."""
        panels = [self.header_panel(), self.progress_panel()]
        if self.evaluation_summary is not None:
            panels.append(self.evaluation_panel())
        elif not self.eval_only:
            panels.append(self.live_metric_panel())
        if not self.eval_only:
            panels.append(self.sparkline_panel())
        return Group(*panels)

    def header_panel(self) -> Panel:
        """Build immutable run identity and dataset details."""
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(Text("model", style="dim"), self.model_id)
        table.add_row(Text("device", style="dim"), self.device_name)
        table.add_row(Text("dataset", style="dim"), self.args.dataset)
        table.add_row(Text("repo", style="dim"), self.repo_id)
        table.add_row(
            Text("labels", style="dim"),
            f"{self.screen_count or '-'} screen / {self.safety_count or '-'} safety",
        )
        table.add_row(
            Text("samples", style="dim"),
            f"train {self.train_count or '-'} / val {self.val_count or '-'} / test {self.test_count or '-'}",
        )
        table.add_row(
            Text("batch", style="dim"),
            (
                f"{self.args.micro_batch_size} x accum {self.args.grad_accum_steps or 'auto'}"
                f" / workers {self.args.workers}"
            ),
        )
        table.add_row(Text("image", style="dim"), f"{self.args.image_size}px max side")
        return Panel(table, title="Screenshot Classifier", border_style="cyan")

    def progress_panel(self) -> Panel:
        """Build phase progress, throughput, epoch, and ETA."""
        progress = self.stage_progress()
        eta = self.eta(progress[0], progress[1])
        text = Text()
        text.append(f"{self.stage}  ", style="bold")
        text.append(format_progress_bar(progress[0], progress[1]))
        text.append(f"  {progress[0]}/{progress[1]}")
        if self.epoch > 0:
            text.append(f"  epoch {self.epoch}/{self.args.epochs}")
        if self.images_per_second > 0:
            text.append(f"  {self.images_per_second:.1f} img/s")
        text.append(f"  eta {eta}")
        return Panel(text, border_style="blue")

    def live_metric_panel(self) -> Panel:
        """Build loss and validation metrics accumulated so far."""
        table = Table(expand=True, show_header=True, header_style="bold")
        table.add_column("metric")
        table.add_column("current", justify="right")
        table.add_row("loss", format_metric_value(self.train_loss))
        table.add_row("screen loss", format_metric_value(self.screen_loss))
        table.add_row("safety loss", format_metric_value(self.safety_loss))
        table.add_row("learning rate", format_metric_value(self.learning_rate))
        table.add_row("val score", format_metric_value(self.score))
        table.add_row("best score", self.best_value())
        table.add_row("screen f1", format_metric_value(self.screen_macro_f1))
        table.add_row("screen balanced", format_metric_value(self.screen_balanced_accuracy))
        table.add_row("screen top2", format_metric_value(self.screen_top2_accuracy))
        table.add_row("safety f1", format_metric_value(self.safety_macro_f1))
        table.add_row("safety balanced", format_metric_value(self.safety_balanced_accuracy))
        table.add_row("safety top2", format_metric_value(self.safety_top2_accuracy))
        for label, recall in self.safety_recalls.items():
            table.add_row(f"{label} recall", format_metric_value(recall))
        return Panel(table, border_style="green")

    def evaluation_panel(self) -> Panel:
        """Build final torch and ONNX evaluation results."""
        evaluation_summary = self.evaluation_summary
        if evaluation_summary is None:
            return Panel(Text("No final metrics"), title="Final Results", border_style="green")
        table = Table(expand=True, show_header=True, header_style="bold")
        table.add_column("model")
        table.add_column("test")
        table.add_column("head")
        table.add_column("accuracy", justify="right")
        table.add_column("balanced acc", justify="right")
        table.add_column("macro f1", justify="right")
        table.add_column("top2", justify="right")
        table.add_column("images", justify="right")
        for row in build_evaluation_rows(evaluation_summary):
            table.add_row(*row)
        timing = collect_timing_values(evaluation_summary)
        if not timing:
            return Panel(table, title="Final Results", border_style="green")
        return Panel(Group(table, Text(""), build_timing_table(timing)), title="Final Results", border_style="green")

    def sparkline_panel(self) -> Panel:
        """Build the rolling training-loss trend."""
        return Panel(
            Text("loss trend  " + format_sparkline(self.loss_points), style="magenta"),
            border_style="magenta",
        )

    def stage_progress(self) -> tuple[int, int]:
        """Return completed and total work for the active phase."""
        if self.stage == "train":
            return self.batch, self.batches
        if self.eval_total > 0:
            return self.eval_done, self.eval_total
        return self.stage_done, self.stage_total

    def eta(self, done: int, total: int) -> str:
        """Estimate remaining phase duration from completed work."""
        if done <= 0 or done >= total:
            return "--:--:--:---"
        elapsed = time.perf_counter() - self.phase_started_at
        remaining = elapsed * (total - done) / done
        return format_elapsed_time(remaining * 1000)

    def best_value(self) -> str:
        """Format the best validation score with its epoch."""
        if self.best_score is None:
            return "-"
        epoch = "-" if self.best_epoch is None else str(self.best_epoch)
        return f"{self.best_score:.4f} @ {epoch}"
