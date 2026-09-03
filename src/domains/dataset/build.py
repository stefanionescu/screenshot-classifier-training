"""Build, validate, and atomically publish a local dataset artifact."""

from __future__ import annotations

import os
import json
import shutil
import asyncio
import tempfile
from pathlib import Path
from dataclasses import asdict
from typing import TYPE_CHECKING
from src.shared.output import json_safe
from src.config.dataset import DATASET_CONFIG
from src.domains.dataset.card import build_dataset_card
from src.shared.paths import confined_path, output_path
from src.domains.dataset.collect import collect_dataset_entries
from src.domains.dataset.manifest import write_dataset_manifest
from src.domains.dataset.webdataset import write_dataset_shards
from src.domains.dataset.validate import validate_dataset_output
from src.state.dataset import DatasetBuildResult, DatasetSummary

if TYPE_CHECKING:
    from collections.abc import Mapping
    from src.state.contracts import DatasetSplit
    from src.state.dataset import DatasetBuildOptions, DatasetEntry, DatasetProgressCallback, DatasetRejectedImage


def _create_summary(
    entries: list[DatasetEntry],
    rejected: list[DatasetRejectedImage],
    shard_counts: Mapping[DatasetSplit, int],
) -> DatasetSummary:
    """Build label, rejection, and shard counts for the staged artifact."""
    screen_counts: dict[str, int] = {}
    safety_counts = dict.fromkeys(DATASET_CONFIG["source"]["safety_values"], 0)
    for entry in entries:
        screen_counts[entry.screen] = screen_counts.get(entry.screen, 0) + 1
        safety_counts[entry.safety] += 1
    return DatasetSummary(
        total_images=len(entries),
        rejected_images=len(rejected),
        shard_counts={split: shard_counts[split] for split in DATASET_CONFIG["split"]["names"]},
        screen=sorted(screen_counts),
        safety=list(DATASET_CONFIG["source"]["safety_values"]),
        screen_counts={screen: screen_counts[screen] for screen in sorted(screen_counts)},
        safety_counts=safety_counts,
    )


def _secure_sibling(parent: Path, prefix: str) -> Path:
    """Create a private temporary sibling directory."""
    return Path(os.path.normpath(tempfile.mkdtemp(prefix=prefix, dir=parent)))


def _replace_directory(staging_dir: Path, target_dir: Path) -> None:
    """Swap a validated staging directory into place with rollback."""
    target_dir = Path(os.path.normpath(output_path(target_dir, is_root_allowed=False)))
    staging_dir = Path(
        os.path.normpath(
            confined_path(
                target_dir.parent,
                staging_dir,
                is_root_allowed=False,
                is_existing_required=True,
            ),
        ),
    )
    backup_dir = Path(os.path.normpath(_secure_sibling(target_dir.parent, f".{target_dir.name}-backup-")))
    backup_dir.rmdir()
    had_target = target_dir.exists()
    if had_target:
        if not target_dir.is_dir():
            msg = "dataset output target must be a directory."
            raise ValueError(msg)
        target_dir.replace(backup_dir)
    try:
        staging_dir.replace(target_dir)
    except BaseException:
        if had_target and backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _remove_staging(staging_dir: Path) -> None:
    """Remove a failed staging directory confined to project output."""
    staging_dir = Path(os.path.normpath(output_path(staging_dir, is_root_allowed=False)))
    if staging_dir.exists():
        shutil.rmtree(staging_dir)


async def build_dataset(
    options: DatasetBuildOptions,
    on_progress: DatasetProgressCallback | None = None,
) -> DatasetBuildResult:
    """Build a complete dataset in staging and atomically publish it."""
    target_dir = output_path(options.output_dir, is_root_allowed=False)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = _secure_sibling(target_dir.parent, f".{target_dir.name}-staging-")
    try:
        collection = await collect_dataset_entries(options, on_progress)
        shards_dir = staging_dir / DATASET_CONFIG["output"]["shards_dir"]
        shards_dir.mkdir()
        shard_result = await asyncio.to_thread(
            write_dataset_shards,
            collection.entries,
            shards_dir,
            on_progress,
        )
        await asyncio.to_thread(
            write_dataset_manifest,
            shard_result.manifest_rows,
            staging_dir,
            on_progress,
        )
        summary = _create_summary(collection.entries, collection.rejected, shard_result.shard_counts)
        card = await asyncio.to_thread(build_dataset_card, summary, options)
        (staging_dir / DATASET_CONFIG["output"]["readme_file"]).write_text(card, encoding="utf-8")
        rejection_content = "".join(f"{json.dumps(asdict(item))}\n" for item in collection.rejected)
        (staging_dir / DATASET_CONFIG["output"]["rejected_jsonl_file"]).write_text(
            rejection_content,
            encoding="utf-8",
        )
        metadata_content = json.dumps(json_safe(summary), indent=2) + "\n"
        (staging_dir / DATASET_CONFIG["output"]["metadata_file"]).write_text(
            metadata_content,
            encoding="utf-8",
        )
        await asyncio.to_thread(validate_dataset_output, staging_dir)
        await asyncio.to_thread(_replace_directory, staging_dir, target_dir)
        return DatasetBuildResult(output_dir=str(target_dir), summary=summary)
    except BaseException:
        _remove_staging(staging_dir)
        raise
