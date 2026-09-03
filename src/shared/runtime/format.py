"""Formatting helpers for CLI output."""

from __future__ import annotations


def format_elapsed_time(milliseconds: float) -> str:
    """Format elapsed milliseconds as HH:MM:SS:mmm."""
    total_milliseconds = max(0, int(milliseconds))
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    seconds = (total_milliseconds % 60_000) // 1000
    remaining_milliseconds = total_milliseconds % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{remaining_milliseconds:03d}"
