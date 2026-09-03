"""Rich CLI progress bar."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.shared.runtime.elapsed import ElapsedColumn
from rich.progress import BarColumn, Progress, TaskID, TextColumn

if TYPE_CHECKING:
    from typing import Self
    from types import TracebackType
    from collections.abc import Callable


class ProgressBar:
    """Progress bar for command-line batch work."""

    def __init__(self, total: int, format_value: Callable[[int], str] | None = None) -> None:
        """Create a progress bar for a known item count."""
        if total < 1:
            msg = "progress total must be a positive integer."
            raise ValueError(msg)
        self._total = total
        self._is_finished = False
        self._format_value = format_value or (str)
        self._progress = Progress(
            TextColumn("{task.percentage:>3.0f}%"),
            BarColumn(),
            TextColumn("{task.fields[completed_value]}/{task.fields[total_value]}"),
            ElapsedColumn(),
            transient=False,
        )
        self._task_id: TaskID = self._progress.add_task(
            "work",
            total=self._total,
            completed_value=self._format_value(0),
            total_value=self._format_value(self._total),
        )
        self._progress.start()

    def __enter__(self) -> Self:
        """Return the active progress bar."""
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Stop terminal rendering when the owning operation exits."""
        self.stop()

    def update(self, completed: int, _total: int | None = None, _item: object | None = None) -> None:
        """Display progress for a completed count."""
        if not 0 <= completed <= self._total:
            msg = f"progress must be between 0 and {self._total}."
            raise ValueError(msg)
        if self._is_finished:
            return
        self._progress.update(
            self._task_id,
            completed=completed,
            completed_value=self._format_value(completed),
        )

    def stop(self) -> None:
        """Stop terminal rendering."""
        if self._is_finished:
            return
        self._progress.stop()
        self._is_finished = True
