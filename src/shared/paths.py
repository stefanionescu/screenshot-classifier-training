"""Stable project path utilities."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"


def confined_path(
    root: Path,
    value: str | Path,
    *,
    is_root_allowed: bool = True,
    is_existing_required: bool = False,
) -> Path:
    """Resolve a path and require it to remain inside an explicit root."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved_root = root.resolve(strict=False)
    resolved = candidate.resolve(strict=is_existing_required)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        msg = f"path must remain inside {resolved_root}: {candidate}"
        raise ValueError(msg) from error
    if not is_root_allowed and relative == Path():
        msg = f"path must be a child of {resolved_root}: {candidate}"
        raise ValueError(msg)
    return resolved


def output_path(
    value: str | Path,
    *,
    is_root_allowed: bool = True,
    is_existing_required: bool = False,
) -> Path:
    """Resolve an absolute, project-relative, or output-relative artifact path."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate if candidate.parts[:1] == (OUTPUT_ROOT.name,) else OUTPUT_ROOT / candidate
    return confined_path(
        OUTPUT_ROOT,
        candidate,
        is_root_allowed=is_root_allowed,
        is_existing_required=is_existing_required,
    )


def format_short_path(file_path: str, segment_count: int = 2) -> str:
    """Format a path by keeping trailing path segments."""
    path = Path(file_path)
    parts = path.parts[-segment_count:]
    return str(Path(*parts)) if parts else str(path)
