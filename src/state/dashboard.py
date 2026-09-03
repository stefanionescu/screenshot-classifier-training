"""Training dashboard state protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.state.metrics import EvaluationMetrics, EvaluationSummary


@dataclass(frozen=True)
class TrainUpdate:
    """Training progress update values."""

    batch: int
    batch_images: int
    train_loss: float
    screen_loss: float
    safety_loss: float
    learning_rate: float


class TrainDashboardProtocol(Protocol):
    """Dashboard operations used by training state consumers."""

    device_name: str
    eval_only: bool
    eval_done: int
    evaluation_summary: EvaluationSummary | None

    def set_stage(self, stage: str, done: int = 0, total: int = 1) -> None:
        """Set the current dashboard stage."""
        raise NotImplementedError(stage, done, total)

    def advance_stage(self) -> None:
        """Advance current dashboard stage progress."""
        raise NotImplementedError

    def set_counts(
        self,
        screen_count: int,
        safety_count: int,
        train_count: int,
        val_count: int,
        test_count: int,
    ) -> None:
        """Set dataset counts."""
        raise NotImplementedError(screen_count, safety_count, train_count, val_count, test_count)

    def start_epoch(self, epoch: int, batches: int, learning_rate: float) -> None:
        """Start an epoch."""
        raise NotImplementedError(epoch, batches, learning_rate)

    def update_train(self, update: TrainUpdate) -> None:
        """Update training metrics."""
        raise NotImplementedError(update)

    def start_eval(self, stage: str, total: int) -> None:
        """Start evaluation progress."""
        raise NotImplementedError(stage, total)

    def set_validation(
        self,
        screen_metrics: EvaluationMetrics,
        safety_metrics: EvaluationMetrics,
        score: float,
        best_score: float,
        best_epoch: int,
    ) -> None:
        """Set validation metrics."""
        raise NotImplementedError(screen_metrics, safety_metrics, score, best_score, best_epoch)

    def refresh(self, *, force: bool = False) -> None:
        """Refresh the dashboard display."""
        raise NotImplementedError(force)
