"""Capture, validate, and restore training random-number state."""

from __future__ import annotations

import torch
import random
import numpy as np
from typing import cast
from src.errors import TrainingError
from src.state.rng import NumpyStateSnapshot, RandomStateSnapshot

type CheckpointMap = dict[str, object]


def random_state() -> CheckpointMap:
    """Capture every random-number generator used by training."""
    numpy_state = cast(
        "tuple[str, np.ndarray[tuple[int], np.dtype[np.uint32]], int, int, float]",
        np.random.get_state(),
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def validate_numpy_state(value: object) -> NumpyStateSnapshot:
    """Validate one persisted NumPy random-number state."""
    if not isinstance(value, dict):
        msg = "checkpoint NumPy random-number state is invalid."
        raise TrainingError(msg)
    numpy_state = cast("dict[object, object]", value)
    if set(numpy_state) != {"name", "keys", "position", "has_gauss", "cached_gaussian"}:
        msg = "checkpoint NumPy random-number fields do not match the supported schema."
        raise TrainingError(msg)
    name = numpy_state.get("name")
    keys = numpy_state.get("keys")
    position = numpy_state.get("position")
    has_gauss = numpy_state.get("has_gauss")
    cached_gaussian = numpy_state.get("cached_gaussian")
    key_values = cast("list[object]", keys) if isinstance(keys, list) else []
    if (
        not isinstance(name, str)
        or not isinstance(keys, list)
        or any(isinstance(key, bool) or not isinstance(key, int) for key in key_values)
        or isinstance(position, bool)
        or not isinstance(position, int)
        or isinstance(has_gauss, bool)
        or not isinstance(has_gauss, int)
        or isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, int | float)
    ):
        msg = "checkpoint NumPy random-number state is invalid."
        raise TrainingError(msg)
    return NumpyStateSnapshot(
        name=name,
        keys=np.asarray(cast("list[int]", key_values), dtype=np.uint32),
        position=position,
        has_gauss=has_gauss,
        cached_gaussian=float(cached_gaussian),
    )


def validate_random_state(value: object) -> RandomStateSnapshot:
    """Validate every persisted random-number state without changing global state."""
    if not isinstance(value, dict):
        msg = "checkpoint random-number state is invalid."
        raise TrainingError(msg)
    state = cast("dict[object, object]", value)
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        msg = "checkpoint random-number fields do not match the supported schema."
        raise TrainingError(msg)
    python_state = state.get("python")
    torch_cpu = state.get("torch_cpu")
    torch_cuda = state.get("torch_cuda")
    cuda_values = cast("list[object]", torch_cuda) if isinstance(torch_cuda, list) else []
    if (
        not isinstance(python_state, tuple)
        or not torch.is_tensor(torch_cpu)
        or not isinstance(torch_cuda, list)
        or any(not torch.is_tensor(item) for item in cuda_values)
    ):
        msg = "checkpoint random-number state is invalid."
        raise TrainingError(msg)
    numpy_state = validate_numpy_state(state.get("numpy"))
    snapshot = RandomStateSnapshot(
        python=cast("tuple[object, ...]", python_state),
        numpy_name=numpy_state.name,
        numpy_keys=numpy_state.keys,
        numpy_position=numpy_state.position,
        numpy_has_gauss=numpy_state.has_gauss,
        numpy_cached_gaussian=numpy_state.cached_gaussian,
        torch_cpu=torch_cpu.cpu(),
        torch_cuda=[item.cpu() for item in cast("list[torch.Tensor]", cuda_values)],
    )
    validate_rng_snapshot(snapshot)
    return snapshot


def validate_rng_snapshot(snapshot: RandomStateSnapshot) -> None:
    """Ask isolated generators to validate restored state shapes and values."""
    previous_python_state = random.getstate()
    try:
        random.setstate(snapshot.python)
        numpy_generator = np.random.RandomState()
        numpy_generator.set_state(
            (
                snapshot.numpy_name,
                snapshot.numpy_keys,
                snapshot.numpy_position,
                snapshot.numpy_has_gauss,
                snapshot.numpy_cached_gaussian,
            ),
        )
        torch.Generator(device="cpu").set_state(snapshot.torch_cpu)
    except (RuntimeError, TypeError, ValueError) as error:
        msg = "checkpoint random-number state is invalid."
        raise TrainingError(msg) from error
    finally:
        random.setstate(previous_python_state)


def restore_random_state(snapshot: RandomStateSnapshot) -> None:
    """Restore a previously validated random-number state."""
    random.setstate(snapshot.python)
    np.random.set_state(
        (
            snapshot.numpy_name,
            snapshot.numpy_keys,
            snapshot.numpy_position,
            snapshot.numpy_has_gauss,
            snapshot.numpy_cached_gaussian,
        ),
    )
    torch.set_rng_state(snapshot.torch_cpu)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.torch_cuda)
