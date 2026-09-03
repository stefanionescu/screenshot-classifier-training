"""Shared image collection helpers."""

from __future__ import annotations

from pathlib import Path
from src.config.image import IMAGE_CONFIG
from src.state.contracts import ImageItem
from src.shared.paths import IMAGE_ROOT, confined_path
from src.shared.images.path import get_category_from_path, validated_image_path

EXCLUDED_DIRS = set(IMAGE_CONFIG["excluded_dirs"])


def collect_path_images(input_paths: list[str] | tuple[str, ...]) -> list[ImageItem]:
    """Collect images from explicit files or directories."""
    items: dict[str, ImageItem] = {}
    for input_path in input_paths:
        path = Path(validated_image_path(input_path))
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in paths:
            resolved = confined_path(IMAGE_ROOT, candidate, is_existing_required=True)
            relative = resolved.relative_to(IMAGE_ROOT.resolve(strict=False))
            if any(part in EXCLUDED_DIRS for part in relative.parts):
                continue
            if resolved.is_file() and resolved.suffix.lower() in IMAGE_CONFIG["extensions"]:
                full = str(resolved)
                items[full] = ImageItem(image_path=full, category=get_category_from_path(full))
    return list(items.values())
