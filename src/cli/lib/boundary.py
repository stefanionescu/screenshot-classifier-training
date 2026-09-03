"""Shared command error boundary."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from src.errors import TrainingError
from contextlib import contextmanager

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def cli_error_boundary() -> Generator[None, None, None]:
    """Map typed application failures to concise command exit statuses."""
    try:
        yield
    except TrainingError as error:
        sys.stderr.write(f"error: {error}\n")
        raise SystemExit(1) from error
