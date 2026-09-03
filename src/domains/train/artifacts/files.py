"""Atomic file operations for completed training artifacts."""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from src.shared.output import json_safe
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from src.state.contracts import JsonValue


class FileDigest(TypedDict):
    """Digest and byte size for one artifact file."""

    sha256: str
    size: int


def read_json(path: Path) -> JsonValue:
    """Read and validate one JSON value."""
    return json_safe(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: object) -> None:
    """Validate and write one JSON value."""
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text(target, json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    """Write text."""
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temp_path = Path(handle.name).resolve(strict=True)
            handle.write(value)
        temp_path.parent.relative_to(target.parent)
        temp_path.replace(target)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def file_digest(path: Path) -> FileDigest:
    """Return the SHA-256 digest and byte size for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return FileDigest(sha256=digest.hexdigest(), size=path.stat().st_size)
