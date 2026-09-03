"""Training export file checks."""

from __future__ import annotations

import shutil
import filecmp
from pathlib import Path
from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.domains.train.artifacts.paths import repo_root
from src.domains.train.artifacts.files import read_json, write_text
from src.domains.train.artifacts.run_files import train_config_path
from src.config.train.artifacts import (
    TRAIN_ONNX_DIR,
    TRAIN_CHECKPOINTS_DIR,
    TRAIN_CONFIG_FILENAME,
    ONNX_HALF_MODEL_FILENAME,
    CURRENT_CHECKPOINT_FILENAME,
    TRAIN_REQUIRED_EXPORT_PATHS,
    TRAIN_BEST_CHECKPOINT_FILENAME,
    TRAIN_EXPORT_CHECKPOINT_FILENAME,
)

if TYPE_CHECKING:
    from src.state.train.training import TrainArgs


def copy_train_config(run_dir: Path, export_dir: Path) -> None:
    """Copy the training recipe into the export."""
    value = train_config_path(run_dir).read_text(encoding="utf-8")
    write_text(export_dir / TRAIN_CONFIG_FILENAME, value)


def copy_checkpoint(run_dir: Path, export_dir: Path, args: TrainArgs) -> None:
    """Copy the selected checkpoint into the export."""
    checkpoint_filename = TRAIN_BEST_CHECKPOINT_FILENAME
    if args.export_checkpoint == "latest":
        checkpoint_filename = CURRENT_CHECKPOINT_FILENAME
    run_checkpoint_dir = (run_dir.resolve(strict=True) / TRAIN_CHECKPOINTS_DIR).resolve(strict=False)
    export_root = export_dir.resolve(strict=False)
    checkpoint_dir = (export_root / TRAIN_CHECKPOINTS_DIR).resolve(strict=False)
    source = (run_checkpoint_dir / checkpoint_filename).resolve(strict=False)
    checkpoint_dir.relative_to(export_root)
    source.relative_to(run_checkpoint_dir)
    if not source.is_file():
        msg = f"checkpoint does not exist: {source.relative_to(repo_root())}"
        raise TrainingError(msg)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = (checkpoint_dir / TRAIN_EXPORT_CHECKPOINT_FILENAME).resolve(strict=False)
    destination.relative_to(checkpoint_dir)
    shutil.copy2(source, destination)


def export_ready(run_dir: Path, export_dir: Path, args: TrainArgs) -> bool:
    """Return whether export files match the current run."""
    if not export_files_exist(export_dir, args) or not train_config_matches(run_dir, export_dir):
        return False
    checkpoint_filename = TRAIN_BEST_CHECKPOINT_FILENAME
    if args.export_checkpoint == "latest":
        checkpoint_filename = CURRENT_CHECKPOINT_FILENAME
    checkpoint_dir = export_dir / TRAIN_CHECKPOINTS_DIR
    if not checkpoint_dir.is_dir():
        return False
    paths = list(checkpoint_dir.iterdir())
    if len(paths) != 1 or paths[0].name != TRAIN_EXPORT_CHECKPOINT_FILENAME or not paths[0].is_file():
        return False
    source = run_dir / TRAIN_CHECKPOINTS_DIR / checkpoint_filename
    return source.is_file() and filecmp.cmp(source, paths[0], shallow=False)


def export_files_exist(export_dir: Path, args: TrainArgs) -> bool:
    """Return whether required export files exist."""
    paths = [Path(path) for path in TRAIN_REQUIRED_EXPORT_PATHS]
    if args.export == "fp16":
        paths.append(Path(TRAIN_ONNX_DIR) / ONNX_HALF_MODEL_FILENAME)
    return all((export_dir / path).is_file() for path in paths)


def train_config_matches(run_dir: Path, export_dir: Path) -> bool:
    """Return whether the export recipe matches the run recipe."""
    return read_json(train_config_path(run_dir)) == read_json(export_dir / TRAIN_CONFIG_FILENAME)
