"""Cache model inputs and publish completed exports to Hugging Face Hub."""

from __future__ import annotations

import os
import huggingface_hub
from pathlib import Path
from huggingface_hub import HfApi
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
from src.config.train.model import SUPPORTED_MODELS
from huggingface_hub.utils import logging as hf_logging
from src.domains.train.artifacts.paths import repo_root
from huggingface_hub.utils.tqdm import disable_progress_bars
from src.config.train.paths import TRAIN_HUGGINGFACE_CACHE_DIR
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.state.train.training import TrainArgs
    from src.state.train.dashboard import TrainDashboardProtocol

MODEL_ATTRIBUTE_LINES = (
    "*.onnx filter=lfs diff=lfs merge=lfs -text",
    "*.onnx.data filter=lfs diff=lfs merge=lfs -text",
    "*.pt filter=lfs diff=lfs merge=lfs -text",
    "*.safetensors filter=lfs diff=lfs merge=lfs -text",
)
PUSH_STAGE = "push"


def require_hf_token() -> str:
    """Return the configured Hub token or reject an upload request."""
    token = hf_token()
    if token:
        return token
    msg = "HF_TOKEN must be set in the environment or .env before using --push or --push-only."
    raise TrainingError(msg)


def hf_token() -> str | None:
    """Return the token loaded by the CLI environment boundary."""
    return os.environ.get("HF_TOKEN")


def normalize_model_id(model: str) -> str:
    """Normalize model id."""
    model_id = model.removeprefix("hf_hub:")
    if "/" not in model_id:
        model_id = f"timm/{model_id}"
    if model_id not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        msg = f"unsupported --model: {model_id}. Supported models: {supported}"
        raise TrainingError(msg)
    return model_id


def cache_model(model_id: str) -> None:
    """Ensure model cached."""
    cache_dir = repo_root() / TRAIN_HUGGINGFACE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token()
    disable_progress_bars()
    hf_logging.set_verbosity_error()
    read_snapshot: Callable[..., str] = cast("Callable[..., str]", huggingface_hub.snapshot_download)
    try:
        read_snapshot(repo_id=model_id, cache_dir=str(cache_dir), local_files_only=True, token=token)
    except LocalEntryNotFoundError:
        read_snapshot(repo_id=model_id, cache_dir=str(cache_dir), token=token)


def push_export(export_dir: Path, repo_id: str, token: str, *, is_private: bool) -> None:
    """Push export."""
    if not export_dir.is_dir():
        msg = "export directory does not exist. Export the model before uploading it."
        raise TrainingError(msg)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=is_private, exist_ok=True)
    api.update_repo_settings(repo_id=repo_id, repo_type="model", private=is_private)
    if write_model_attributes(export_dir, repo_id, token):
        api.upload_file(
            path_or_fileobj=str(export_dir / ".gitattributes"),
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model storage attributes",
        )
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload phone screen classifier",
        delete_patterns="*",
    )


def publish_training_export(
    args: TrainArgs,
    token: str | None,
    repo_id: str,
    export_dir: Path,
    dashboard: TrainDashboardProtocol,
) -> None:
    """Publish a completed export when upload was requested."""
    if not args.push:
        return
    if token is None:
        msg = "Hugging Face token is required for model upload."
        raise TrainingError(msg)
    dashboard.set_stage(PUSH_STAGE, 0, 1)
    push_export(export_dir, repo_id, token, is_private=not args.public)
    dashboard.advance_stage()


def write_model_attributes(export_dir: Path, repo_id: str, token: str) -> bool:
    """Write model attributes."""
    lines = existing_attribute_lines(repo_id, token)
    stored_lines = tuple(lines)
    for line in MODEL_ATTRIBUTE_LINES:
        if line not in lines:
            lines.append(line)
    (export_dir / ".gitattributes").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tuple(lines) != stored_lines


def existing_attribute_lines(repo_id: str, token: str) -> list[str]:
    """Read the remote model repository's Git attribute rules when present."""
    read_hub_file: Callable[..., str] = cast("Callable[..., str]", huggingface_hub.hf_hub_download)
    try:
        path = Path(read_hub_file(repo_id=repo_id, filename=".gitattributes", repo_type="model", token=token)).resolve(
            strict=True,
        )
    except EntryNotFoundError:
        return []
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()
