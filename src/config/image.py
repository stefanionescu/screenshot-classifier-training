"""Image processing configuration."""

from src.state.image import ImageConfig

IMAGE_CONFIG: ImageConfig = {
    "extensions": (".jpg", ".jpeg", ".png", ".webp"),
    "excluded_dirs": ("UNCATEGORIZED",),
    "images_dir": "dataset",
}
