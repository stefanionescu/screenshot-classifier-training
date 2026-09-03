"""Summarize image decode failures recorded during training."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.domains.train.artifacts.paths import repo_root
from src.state.train.metrics import SkippedImagesSummary
from src.domains.train.artifacts.run_files import skipped_images_path
from src.domains.train.images.skipped.journal import read_skipped_image_row

if TYPE_CHECKING:
    from pathlib import Path


def skipped_images_summary(run_dir: Path) -> SkippedImagesSummary:
    """Summarize skipped-image counts by dataset split."""
    path = skipped_images_path(run_dir)
    counts = {"train": 0, "val": 0, "test": 0}
    total = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = read_skipped_image_row(line)
            if row is None:
                continue
            split = row.get("split")
            if split in counts:
                counts[split] += 1
            total += 1
    return SkippedImagesSummary(
        file=str(path.relative_to(repo_root())),
        total=total,
        counts=counts,
    )
