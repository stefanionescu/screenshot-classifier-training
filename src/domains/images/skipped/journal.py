"""Append skipped-image records without repeatedly scanning the journal."""

from __future__ import annotations

import os
import json
import fcntl
from src.state.training import SkippedImageRow
from typing import TYPE_CHECKING, TextIO, cast
from src.domains.artifacts.paths import repo_root

if TYPE_CHECKING:
    from pathlib import Path
    from src.state.training import Sample

SKIPPED_IMAGE_SCHEMA = 1
type SkippedKey = tuple[object, object, object]
SKIPPED_IMAGE_FIELDS = {
    "error",
    "height",
    "image_member",
    "safety_label",
    "schema",
    "screen_label",
    "split",
    "tar_path",
    "width",
}


class SkippedImageJournal:
    """Maintain a process-local index over an interprocess-safe JSONL journal."""

    def __init__(self, path: Path | None) -> None:
        """Create a journal for an optional destination path."""
        self.path = path
        self._offset = 0
        self._keys: set[SkippedKey] = set()

    def record(self, sample: Sample, error: Exception) -> None:
        """Append a skipped image once across all data-iterator processes."""
        if self.path is None:
            return
        row = skipped_image_row(sample, error)
        row_key = skipped_image_key(row)
        line = json.dumps(row, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            self._read_appended_rows(handle)
            if row_key not in self._keys:
                handle.seek(0, os.SEEK_END)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                self._offset = handle.tell()
                self._keys.add(row_key)
            fcntl.flock(handle, fcntl.LOCK_UN)

    def _read_appended_rows(self, handle: TextIO) -> None:
        """Index only rows written since this process last held the lock."""
        handle.seek(self._offset)
        for line in handle:
            row = read_skipped_image_row(line)
            if row is not None:
                self._keys.add(skipped_image_key(row))
        self._offset = handle.tell()


def skipped_image_row(sample: Sample, error: Exception) -> SkippedImageRow:
    """Build one stable skipped-image record."""
    resolved = sample.tar_path if sample.tar_path.is_absolute() else repo_root() / sample.tar_path
    return SkippedImageRow(
        schema=SKIPPED_IMAGE_SCHEMA,
        split=sample.split,
        tar_path=str(resolved.resolve().relative_to(repo_root())),
        image_member=sample.image_name,
        screen_label=sample.screen_label,
        safety_label=sample.safety_label,
        width=sample.width,
        height=sample.height,
        error=f"{type(error).__name__}: {error}",
    )


def skipped_image_key(row: SkippedImageRow) -> SkippedKey:
    """Return the identity fields for one skipped-image record."""
    return row["split"], row["tar_path"], row["image_member"]


def read_skipped_image_row(line: str) -> SkippedImageRow | None:
    """Parse one complete skipped-image row, ignoring a damaged record."""
    try:
        row: object = json.loads(line)
    except json.JSONDecodeError:
        return None
    values = cast("dict[object, object]", row) if isinstance(row, dict) else {}
    strings = ("tar_path", "image_member", "screen_label", "safety_label", "error")
    dimensions = values.get("width"), values.get("height")
    valid = (
        set(values) == SKIPPED_IMAGE_FIELDS
        and all(isinstance(values.get(key), str) and bool(values.get(key)) for key in strings)
        and values.get("split") in {"train", "val", "test"}
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in dimensions)
        and values.get("schema") == SKIPPED_IMAGE_SCHEMA
    )
    return cast("SkippedImageRow", values) if valid else None
