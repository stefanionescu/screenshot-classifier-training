"""Encode and decode bounded non-pickle training checkpoint archives."""

from __future__ import annotations

import json
import torch
import zipfile
import tempfile
from typing import cast
from pathlib import Path
from src.errors import TrainingError
import safetensors.torch as safetensors_torch

type CheckpointMap = dict[str, object]
type EncodedCheckpointValue = str | int | float | bool | None | list[EncodedCheckpointValue] | dict[str, object]
type TensorMap = dict[str, torch.Tensor]

CHECKPOINT_ARCHITECTURE = "flat-adaptive-v2"
CHECKPOINT_SCHEMA = 2
CHECKPOINT_JSON_NAME = "checkpoint.json"
CHECKPOINT_TENSORS_NAME = "tensors.safetensors"
CHECKPOINT_TENSOR_KEY = "__tensor__"
CHECKPOINT_DICT_KEY = "__dict__"
CHECKPOINT_TUPLE_KEY = "__tuple__"
CHECKPOINT_PAIR_SIZE = 2
MAX_CHECKPOINT_METADATA_BYTES = 16 * 1024 * 1024
REQUIRED_CHECKPOINT_FIELDS = {
    "architecture",
    "best_epoch",
    "best_score",
    "epoch",
    "image_size",
    "model",
    "model_id",
    "optimizer",
    "rng",
    "safety_labels",
    "samplers",
    "scaler",
    "scheduler",
    "schema",
    "screen_labels",
    "train_config_sha256",
}


def write_checkpoint_archive(path: Path, payload: CheckpointMap) -> None:
    """Atomically write JSON metadata and safetensors into a ZIP archive."""
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
            temp_path = Path(handle.name).resolve(strict=True)
        temp_path.parent.relative_to(target.parent)
        tensors: TensorMap = {}
        encoded = {key: encode_checkpoint_value(value, tensors) for key, value in payload.items()}
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(CHECKPOINT_JSON_NAME, json.dumps(encoded, sort_keys=True).encode("utf-8"))
            archive.writestr(CHECKPOINT_TENSORS_NAME, safetensors_torch.save(tensors))
        temp_path.replace(target)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def encode_checkpoint_value(value: object, tensors: TensorMap) -> EncodedCheckpointValue:
    """Encode checkpoint values as JSON plus tensor references."""
    if torch.is_tensor(value):
        name = f"tensor_{len(tensors)}"
        tensors[name] = value.detach().cpu().contiguous()
        result: EncodedCheckpointValue = {CHECKPOINT_TENSOR_KEY: name}
    elif isinstance(value, dict):
        items = cast("dict[object, object]", value)
        result = {
            CHECKPOINT_DICT_KEY: [
                [encode_checkpoint_key(key), encode_checkpoint_value(item, tensors)] for key, item in items.items()
            ],
        }
    elif isinstance(value, list):
        items = cast("list[object]", value)
        result = [encode_checkpoint_value(item, tensors) for item in items]
    elif isinstance(value, tuple):
        items = cast("tuple[object, ...]", value)
        result = {CHECKPOINT_TUPLE_KEY: [encode_checkpoint_value(item, tensors) for item in items]}
    elif value is None or isinstance(value, str | int | float | bool):
        result = value
    else:
        msg = f"checkpoint value is not serializable: {type(value).__name__}"
        raise TrainingError(msg)
    return result


def encode_checkpoint_key(value: object) -> str | int:
    """Encode a supported checkpoint mapping key."""
    if isinstance(value, bool):
        msg = "checkpoint Boolean keys are not supported."
        raise TrainingError(msg)
    if isinstance(value, str | int):
        return value
    msg = f"checkpoint key is not serializable: {type(value).__name__}"
    raise TrainingError(msg)


def read_checkpoint_archive(path: Path, run_device: torch.device) -> CheckpointMap:
    """Read and validate a non-pickle checkpoint archive."""
    encoded = read_checkpoint_metadata(path)
    validate_checkpoint_envelope(encoded)
    tensors = read_checkpoint_tensors(path)
    values = cast("dict[object, object]", encoded)
    checkpoint = {
        key: decode_checkpoint_value(value, tensors, run_device)
        for key, value in values.items()
        if isinstance(key, str)
    }
    validate_checkpoint_envelope(checkpoint)
    return checkpoint


def read_checkpoint_metadata(path: Path) -> dict[str, object]:
    """Read bounded JSON metadata without loading checkpoint tensors."""
    if not path.is_file():
        msg = "checkpoint does not exist. Select an existing checkpoint archive."
        raise TrainingError(msg)
    try:
        with zipfile.ZipFile(path) as archive:
            validate_checkpoint_members(archive)
            metadata_fields = archive.getinfo(CHECKPOINT_JSON_NAME)
            if metadata_fields.file_size > MAX_CHECKPOINT_METADATA_BYTES:
                msg = "checkpoint metadata exceeds the supported size."
                raise TrainingError(msg)
            encoded = json.loads(archive.read(CHECKPOINT_JSON_NAME).decode("utf-8"))
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        msg = "checkpoint archive is corrupt or unsupported."
        raise TrainingError(msg) from error
    if not isinstance(encoded, dict):
        msg = "checkpoint metadata is invalid."
        raise TrainingError(msg)
    values = cast("dict[object, object]", encoded)
    if any(not isinstance(key, str) for key in values):
        msg = "checkpoint metadata keys must be strings."
        raise TrainingError(msg)
    return cast("dict[str, object]", encoded)


def validate_checkpoint_members(archive: zipfile.ZipFile) -> None:
    """Require the exact regular-file archive member set."""
    if set(archive.namelist()) != {CHECKPOINT_JSON_NAME, CHECKPOINT_TENSORS_NAME}:
        msg = "checkpoint archive members do not match the supported schema."
        raise TrainingError(msg)
    if any(info.is_dir() for info in archive.infolist()):
        msg = "checkpoint archive members must be regular files."
        raise TrainingError(msg)


def read_checkpoint_tensors(path: Path) -> TensorMap:
    """Read safetensors after metadata validation succeeds."""
    try:
        with zipfile.ZipFile(path) as archive:
            validate_checkpoint_members(archive)
            return safetensors_torch.load(archive.read(CHECKPOINT_TENSORS_NAME))
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        msg = "checkpoint tensor archive is corrupt or unsupported."
        raise TrainingError(msg) from error


def validate_checkpoint_envelope(checkpoint: CheckpointMap) -> None:
    """Validate checkpoint fields and schema before domain-specific loading."""
    if set(checkpoint) != REQUIRED_CHECKPOINT_FIELDS:
        msg = "checkpoint fields do not match the supported schema."
        raise TrainingError(msg)
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        msg = "checkpoint schema is not supported."
        raise TrainingError(msg)


def read_checkpoint_epoch(path: Path) -> int:
    """Read the latest completed epoch without constructing training state."""
    encoded = read_checkpoint_metadata(path)
    validate_checkpoint_envelope(encoded)
    epoch = encoded.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        msg = "checkpoint has an invalid epoch."
        raise TrainingError(msg)
    return epoch


def decode_checkpoint_value(value: object, tensors: TensorMap, run_device: torch.device) -> object:
    """Decode a checkpoint JSON value."""
    if isinstance(value, list):
        items = cast("list[object]", value)
        result: object = [decode_checkpoint_value(item, tensors, run_device) for item in items]
    elif isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        result = decode_checkpoint_record(mapping, tensors, run_device)
    elif value is None or isinstance(value, str | int | float | bool):
        result = value
    else:
        msg = "checkpoint metadata contains an invalid value."
        raise TrainingError(msg)
    return result


def decode_checkpoint_record(
    mapping: dict[object, object],
    tensors: TensorMap,
    run_device: torch.device,
) -> object:
    """Decode a typed checkpoint metadata record."""
    if set(mapping) == {CHECKPOINT_TENSOR_KEY}:
        return read_checkpoint_tensor(mapping, tensors, run_device)
    if set(mapping) == {CHECKPOINT_DICT_KEY}:
        return decode_checkpoint_dict(mapping[CHECKPOINT_DICT_KEY], tensors, run_device)
    if set(mapping) == {CHECKPOINT_TUPLE_KEY}:
        tuple_items = mapping[CHECKPOINT_TUPLE_KEY]
        if not isinstance(tuple_items, list):
            msg = "checkpoint tuple metadata is invalid."
            raise TrainingError(msg)
        items = cast("list[object]", tuple_items)
        return tuple(decode_checkpoint_value(item, tensors, run_device) for item in items)
    msg = "checkpoint metadata contains an unknown record."
    raise TrainingError(msg)


def read_checkpoint_tensor(
    mapping: dict[object, object],
    tensors: TensorMap,
    run_device: torch.device,
) -> torch.Tensor:
    """Read one tensor referenced by checkpoint metadata."""
    name = mapping[CHECKPOINT_TENSOR_KEY]
    if not isinstance(name, str) or name not in tensors:
        msg = "checkpoint tensor metadata is invalid."
        raise TrainingError(msg)
    return tensors[name].to(run_device)


def decode_checkpoint_dict(value: object, tensors: TensorMap, run_device: torch.device) -> dict[object, object]:
    """Decode a checkpoint mapping."""
    if not isinstance(value, list):
        msg = "checkpoint mapping metadata is invalid."
        raise TrainingError(msg)
    result: dict[object, object] = {}
    pairs = cast("list[object]", value)
    for pair in pairs:
        if not isinstance(pair, list):
            msg = "checkpoint mapping entry is invalid."
            raise TrainingError(msg)
        items = cast("list[object]", pair)
        if len(items) != CHECKPOINT_PAIR_SIZE:
            msg = "checkpoint mapping entry is invalid."
            raise TrainingError(msg)
        key = decode_checkpoint_key(items[0])
        result[key] = decode_checkpoint_value(items[1], tensors, run_device)
    return result


def decode_checkpoint_key(value: object) -> str | int:
    """Decode a checkpoint mapping key."""
    if isinstance(value, bool):
        msg = "checkpoint Boolean keys are not supported."
        raise TrainingError(msg)
    if isinstance(value, str | int):
        return value
    msg = "checkpoint mapping key is invalid."
    raise TrainingError(msg)
