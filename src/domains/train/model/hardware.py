"""Describe CPU hardware used in evaluation reports."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def cpu_label() -> str:
    """Format the CPU model and available logical-core count."""
    name = cpu_name()
    count = os.cpu_count()
    if count is None:
        return name
    return f"{name} ({count} logical cores)"


def cpu_name() -> str:
    """Return the most specific CPU model available on this platform."""
    if platform.system() == "Linux":
        value = linux_cpu_name()
        if value:
            return value
    value = platform.processor() or platform.machine()
    return value or "unknown CPU"


def linux_cpu_name() -> str | None:
    """Read the Linux processor model name when procfs exposes it."""
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "model name" and value.strip():
            return value.strip()
    return None
