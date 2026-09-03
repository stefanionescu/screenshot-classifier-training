"""Stage and validate complete deployable model exports."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast
from dataclasses import dataclass, replace
import safetensors.torch as safetensors_torch
from src.shared.paths import OUTPUT_ROOT, confined_path
from src.domains.train.artifacts.files import write_json
from src.domains.train.images.preprocess import build_preprocess_spec
from src.domains.train.model.onnx.export import export_onnx, test_onnx_export
from src.domains.train.model.onnx.half_precision import write_half_precision_model
from src.state.train.export import ExportLabelHead, ExportLabels, ExportModelConfig, ExportPreprocess
from src.config.train.artifacts import (
    TRAIN_ONNX_DIR,
    TRAIN_MODEL_NAME,
    ONNX_OUTPUT_NAMES,
    ONNX_MODEL_FILENAME,
    TRAIN_INFERENCE_DIR,
    TRAIN_LABELS_FILENAME,
    ONNX_HALF_MODEL_FILENAME,
    TRAIN_MODEL_ARCHITECTURE,
    TRAIN_PREPROCESS_FILENAME,
    TRAIN_MODEL_CONFIG_FILENAME,
    TRAIN_MODEL_WEIGHTS_FILENAME,
    TRAIN_PYTHON_INFERENCE_FILENAME,
)

if TYPE_CHECKING:
    import torch
    from src.state.train.export import ModelExport
    from collections.abc import Callable, Generator

INFERENCE_SOURCE = Path(__file__).with_name("inference.py")


@dataclass(frozen=True)
class ExportTransaction:
    """Paths owned by one atomic export transaction."""

    target: Path
    staging: Path
    backup: Path


def _export_transaction(target: Path) -> ExportTransaction:
    """Create adjacent staging and reserved backup paths."""
    target = Path(os.path.normpath(confined_path(OUTPUT_ROOT, target, is_root_allowed=False)))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        os.path.normpath(
            Path(tempfile.mkdtemp(prefix=f".{target.name}-staging-", dir=target.parent)).resolve(strict=True),
        ),
    )
    try:
        backup = Path(
            os.path.normpath(
                Path(tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=target.parent)).resolve(strict=True),
            ),
        )
        backup.rmdir()
    except BaseException:
        shutil.rmtree(staging)
        raise
    return ExportTransaction(target, staging, backup)


def _prepare_staging(
    transaction: ExportTransaction,
    model_export: ModelExport,
    *,
    is_existing_copied: bool,
) -> None:
    """Populate an export staging directory from source or an existing export."""
    if is_existing_copied:
        if not transaction.target.is_dir():
            msg = "existing model export does not exist. Export the model before updating it."
            raise FileNotFoundError(msg)
        shutil.copytree(transaction.target, transaction.staging, dirs_exist_ok=True)
        return
    build_export(replace(model_export, export_dir=transaction.staging))


def _publish_staging(transaction: ExportTransaction) -> None:
    """Atomically publish a staged export and restore the old target on swap failure."""
    target = Path(os.path.normpath(confined_path(OUTPUT_ROOT, transaction.target, is_root_allowed=False)))
    staging = Path(os.path.normpath(confined_path(OUTPUT_ROOT, transaction.staging, is_root_allowed=False)))
    backup = Path(os.path.normpath(confined_path(OUTPUT_ROOT, transaction.backup, is_root_allowed=False)))
    if target.exists():
        target.replace(backup)
    try:
        staging.replace(target)
    except OSError:
        if backup.exists():
            backup.replace(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


@contextmanager
def staged_model_export(
    model_export: ModelExport,
    *,
    is_existing_copied: bool = False,
) -> Generator[Path, None, None]:
    """Yield a complete staging directory and publish it atomically on success."""
    transaction = _export_transaction(model_export.export_dir.resolve(strict=False))
    try:
        _prepare_staging(transaction, model_export, is_existing_copied=is_existing_copied)
        yield transaction.staging
    except BaseException:
        if transaction.staging.exists():
            shutil.rmtree(transaction.staging)
        raise
    else:
        _publish_staging(transaction)


def build_export(model_export: ModelExport) -> None:
    """Build export."""
    model = model_export.model
    export_dir = model_export.export_dir
    model_id = model_export.model_id
    labels = model_export.labels
    image_size = model_export.image_size
    args = model_export.args
    export_dir.mkdir(parents=True, exist_ok=True)
    save_weights: Callable[[dict[str, torch.Tensor], Path], None] = cast(
        "Callable[[dict[str, torch.Tensor], Path], None]",
        safetensors_torch.save_file,
    )
    save_weights(model.state_dict(), export_dir / TRAIN_MODEL_WEIGHTS_FILENAME)
    onnx_dir = export_dir / TRAIN_ONNX_DIR
    onnx_dir.mkdir(parents=True, exist_ok=True)
    preprocess = build_preprocess_spec(model_id)
    export_onnx(model, onnx_dir / ONNX_MODEL_FILENAME, image_size)
    test_onnx_export(onnx_dir / ONNX_MODEL_FILENAME, len(labels.screen), len(labels.safety), image_size)
    if args.export == "fp16":
        write_half_precision_model(onnx_dir)
        test_onnx_export(onnx_dir / ONNX_HALF_MODEL_FILENAME, len(labels.screen), len(labels.safety), image_size)
    write_json(
        export_dir / TRAIN_PREPROCESS_FILENAME,
        ExportPreprocess(
            resize_longest_side_px=image_size,
            preserve_aspect_ratio=True,
            crop=False,
            stretch=False,
            horizontal_flip=False,
            mean=list(preprocess.mean),
            std=list(preprocess.std),
        ),
    )
    write_json(
        export_dir / TRAIN_MODEL_CONFIG_FILENAME,
        ExportModelConfig(
            name=TRAIN_MODEL_NAME,
            architecture=TRAIN_MODEL_ARCHITECTURE,
            model=model_id,
            outputs=list(ONNX_OUTPUT_NAMES),
            screen_labels=labels.screen,
            safety_labels=labels.safety,
            resize_longest_side_px=image_size,
        ),
    )
    write_inference_files(export_dir, labels.screen, labels.safety)


def write_inference_files(export_dir: Path, screen_labels: list[str], safety_labels: list[str]) -> None:
    """Write inference files."""
    export_root = export_dir.resolve(strict=False)
    inference_dir = (export_root / TRAIN_INFERENCE_DIR).resolve(strict=False)
    inference_dir.relative_to(export_root)
    inference_dir.mkdir(parents=True)
    source_path = INFERENCE_SOURCE.resolve(strict=True)
    if not source_path.is_file():
        raise FileNotFoundError(TRAIN_PYTHON_INFERENCE_FILENAME)
    inference_path = (inference_dir / TRAIN_PYTHON_INFERENCE_FILENAME).resolve(strict=False)
    inference_path.relative_to(inference_dir)
    shutil.copy2(source_path, inference_path)
    write_json(
        inference_dir / TRAIN_LABELS_FILENAME,
        ExportLabels(
            screen=ExportLabelHead(
                labels=screen_labels,
                label_to_id={label: index for index, label in enumerate(screen_labels)},
            ),
            safety=ExportLabelHead(
                labels=safety_labels,
                label_to_id={label: index for index, label in enumerate(safety_labels)},
            ),
        ),
    )
