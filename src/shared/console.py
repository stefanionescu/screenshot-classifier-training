"""Console stream output."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

type ConsoleField = tuple[str, str]


def write_lines(messages: Iterable[str]) -> None:
    """Write multiple lines to standard output."""
    sys.stdout.write("".join(f"{message}\n" for message in messages))


def write_command_fields(title: str | None, fields: Iterable[ConsoleField]) -> None:
    """Write aligned command metadata or result fields."""
    rows = tuple(fields)
    label_width = max(len(label) for label, _value in rows)
    lines = [""]
    if title is not None:
        lines.append(f"  {title}:")
    lines.extend(f"  {label:<{label_width}}  {value}" for label, value in rows)
    write_lines((*lines, ""))
