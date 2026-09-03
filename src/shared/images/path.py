"""Image path helpers."""

from __future__ import annotations

from pathlib import Path
from src.shared.paths import IMAGE_ROOT, PROJECT_ROOT, confined_path


def validated_image_path(input_path: str) -> str:
    """Resolve a user-supplied path inside the image corpus."""
    path = Path(input_path)
    if not path.is_absolute():
        project_candidate = (PROJECT_ROOT / path).resolve(strict=False)
        try:
            project_candidate.relative_to(IMAGE_ROOT.resolve(strict=False))
            path = project_candidate
        except ValueError:
            path = IMAGE_ROOT / path
    return str(confined_path(IMAGE_ROOT, path, is_existing_required=True))


def get_category_from_path(image_path: str) -> str:
    """Return the category path relative to the image corpus."""
    relative = confined_path(IMAGE_ROOT, image_path, is_existing_required=True).relative_to(
        IMAGE_ROOT.resolve(strict=False),
    )
    return relative.parent.as_posix()
