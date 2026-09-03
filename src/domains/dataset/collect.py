"""Collect and validate classified images for dataset export."""

from __future__ import annotations

import asyncio
import hashlib
import warnings
from PIL import Image
from pathlib import Path
from typing import TYPE_CHECKING, cast
from src.config.dataset import DATASET_CONFIG
from src.shared.images.collect import collect_path_images
from src.shared.runtime.concurrency import run_concurrent
from src.state.dataset import (
    DatasetEntry,
    DatasetCandidate,
    DatasetCollection,
    DatasetProgressEvent,
    DatasetRejectedImage,
)

if TYPE_CHECKING:
    from src.state.contracts import DatasetSafety, DatasetSplit, ImageItem
    from src.state.dataset import DatasetBuildOptions, DatasetProgressCallback

SAFETY_VALUES = frozenset(DATASET_CONFIG["source"]["safety_values"])


def _category_path(category: str) -> str:
    """Return a source-relative category path."""
    return category.replace("\\", "/")


def _screen(category_path: str) -> str:
    """Derive the screen label from a category path."""
    segments = category_path.split("/")
    primary = segments[0]
    secondary = segments[1] if len(segments) > 1 else ""
    if primary in {"message", "phone"} and secondary:
        return f"{primary}-{secondary}"
    return primary


def _safety(category_path: str) -> DatasetSafety:
    """Derive and validate the safety label from a category path."""
    value = category_path.rsplit("/", maxsplit=1)[-1]
    if value in {"safe", "hot", "nsfw", "forbidden"}:
        return cast("DatasetSafety", value)
    msg = "dataset source is missing a safety-label directory."
    raise ValueError(msg)


def _content_sha256(path: Path) -> str:
    """Hash one source image without loading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inspect_image(item: ImageItem) -> DatasetCandidate | DatasetRejectedImage:
    """Validate one image and return metadata or an explicit rejection."""
    category_path = _category_path(item.category)
    source_path = f"{DATASET_CONFIG['source']['path']}/{category_path}/{Path(item.image_path).name}"
    path = Path(item.image_path)
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        if records:
            return DatasetRejectedImage(source_path, "Image decoder emitted a warning.")
        if width <= 0 or height <= 0:
            return DatasetRejectedImage(source_path, "Image dimensions must be positive.")
        category_path = _category_path(item.category)
        return DatasetCandidate(
            image_path=str(path),
            source_path=source_path,
            screen=_screen(category_path),
            safety=_safety(category_path),
            width=width,
            height=height,
            byte_size=path.stat().st_size,
            content_sha256=_content_sha256(path),
        )
    except (OSError, ValueError):
        return DatasetRejectedImage(source_path, "Image could not be decoded or classified from its source path.")


def _held_out_counts(size: int, options: DatasetBuildOptions) -> tuple[int, int]:
    """Allocate held-out counts while retaining at least one training item."""
    if size < DATASET_CONFIG["split"]["min_bucket_size"]:
        return 0, 0
    val_count = max(1, int(size * options.val_percent / 100)) if options.val_percent > 0 else 0
    test_count = max(1, int(size * options.test_percent / 100)) if options.test_percent > 0 else 0
    available = size - 1
    while val_count + test_count > available:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
    return val_count, test_count


def _select_split(index: int, val_count: int, test_count: int) -> DatasetSplit:
    """Return the split assigned to a bucket-local item index."""
    if index < val_count:
        return "val"
    if index < val_count + test_count:
        return "test"
    return "train"


def _assign_splits(candidates: list[DatasetCandidate], options: DatasetBuildOptions) -> dict[str, DatasetSplit]:
    """Assign deterministic splits within screen-and-safety buckets."""
    buckets: dict[tuple[str, DatasetSafety], list[DatasetCandidate]] = {}
    for candidate in candidates:
        buckets.setdefault((candidate.screen, candidate.safety), []).append(candidate)
    assignments: dict[str, DatasetSplit] = {}
    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda candidate: (candidate.content_sha256, candidate.source_path))
        val_count, test_count = _held_out_counts(len(ordered), options)
        for index, candidate in enumerate(ordered):
            assignments[candidate.content_sha256] = _select_split(index, val_count, test_count)
    return assignments


def _deduplicate(
    candidates: list[DatasetCandidate],
    rejected: list[DatasetRejectedImage],
) -> list[DatasetCandidate]:
    """Keep one stable source path for each content hash."""
    unique: list[DatasetCandidate] = []
    first_source_by_hash: dict[str, str] = {}
    for candidate in sorted(candidates, key=lambda item: item.source_path):
        first_source = first_source_by_hash.get(candidate.content_sha256)
        if first_source is not None:
            rejected.append(
                DatasetRejectedImage(
                    candidate.source_path,
                    f"Duplicate image content already represented by {first_source}.",
                ),
            )
            continue
        first_source_by_hash[candidate.content_sha256] = candidate.source_path
        unique.append(candidate)
    return unique


async def collect_dataset_entries(
    options: DatasetBuildOptions,
    on_progress: DatasetProgressCallback | None = None,
) -> DatasetCollection:
    """Collect, validate, deduplicate, and split classified images."""
    items = await asyncio.to_thread(collect_path_images, [DATASET_CONFIG["source"]["path"]])
    ordered_items = sorted(items, key=lambda item: item.image_path)

    def report(completed: int, total: int, _item: ImageItem) -> None:
        """Forward collection progress when a callback is configured."""
        if on_progress is not None:
            on_progress(DatasetProgressEvent("entries", completed, total))

    if on_progress is not None and ordered_items:
        on_progress(DatasetProgressEvent("entries", 0, len(ordered_items)))
    inspected = await run_concurrent(
        ordered_items,
        lambda item, _index: asyncio.to_thread(_inspect_image, item),
        DATASET_CONFIG["collection_concurrency"],
        report,
    )
    candidates = [item for item in inspected if isinstance(item, DatasetCandidate)]
    rejected = [item for item in inspected if isinstance(item, DatasetRejectedImage)]
    candidates = _deduplicate(candidates, rejected)
    if not candidates:
        msg = "dataset build found no valid, unique source images."
        raise ValueError(msg)
    assignments = _assign_splits(candidates, options)
    entries = [
        DatasetEntry(
            prefix=str(index).zfill(DATASET_CONFIG["output"]["sample_prefix_width"]),
            image_path=candidate.image_path,
            source_path=candidate.source_path,
            screen=candidate.screen,
            split=assignments[candidate.content_sha256],
            safety=candidate.safety,
            width=candidate.width,
            height=candidate.height,
            byte_size=candidate.byte_size,
            content_sha256=candidate.content_sha256,
        )
        for index, candidate in enumerate(sorted(candidates, key=lambda item: item.source_path))
    ]
    return DatasetCollection(entries=entries, rejected=sorted(rejected, key=lambda item: item.source_path))
