"""Image configuration contracts."""

from __future__ import annotations

from typing import TypedDict


class ImageConfig(TypedDict):
    """Image decoding and resize configuration."""

    extensions: tuple[str, ...]
    excluded_dirs: tuple[str, ...]
    images_dir: str
