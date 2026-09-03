"""Validate and construct dataset, run, and Hub artifact paths."""

from __future__ import annotations

from pathlib import Path
from src.errors import TrainingError
from src.config.train.model import ORG_NAME


def repo_root() -> Path:
    """Return the installed project root that owns runtime artifacts."""
    return Path(__file__).resolve().parents[4]


def safe_name(value: str, flag: str) -> str:
    """Validate a command value before using it as a path component."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not value:
        msg = f"{flag} must not be empty."
        raise TrainingError(msg)
    if any(character not in allowed for character in value):
        msg = f"{flag} may only contain letters, numbers, dots, underscores, and hyphens."
        raise TrainingError(msg)
    return value


def validated_dataset_path(value: str) -> Path:
    """Resolve a training dataset confined to the repository dataset tree."""
    root = repo_root()
    dataset_path = (root / value).resolve()
    dataset_root = (root / "dataset").resolve()
    try:
        dataset_path.relative_to(dataset_root)
    except ValueError as error:
        msg = "--dataset must point inside the repo dataset/ directory."
        raise TrainingError(msg) from error
    if not (dataset_path / "data").is_dir():
        msg = "--dataset must contain the required data shards."
        raise TrainingError(msg)
    return dataset_path


def target_repo_id(repo: str | None, output_dir: str) -> str:
    """Resolve and constrain the destination model repository identifier."""
    name = repo or output_dir
    if "/" in name:
        if not name.startswith(f"{ORG_NAME}/"):
            msg = f"--repo must target the {ORG_NAME}/ org."
            raise TrainingError(msg)
        return name
    return f"{ORG_NAME}/{safe_name(name, '--repo')}"
