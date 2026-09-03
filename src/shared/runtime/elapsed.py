"""Elapsed-time progress display."""

from __future__ import annotations

from rich.text import Text
from rich.progress import ProgressColumn, Task
from src.shared.runtime.format import format_elapsed_time


class ElapsedColumn(ProgressColumn):
    """Display progress elapsed time to millisecond precision."""

    def render(self, task: Task) -> Text:
        """Render the current task duration with millisecond precision."""
        elapsed_seconds = task.elapsed or 0.0
        return Text(format_elapsed_time(elapsed_seconds * 1000), style="progress.elapsed")
