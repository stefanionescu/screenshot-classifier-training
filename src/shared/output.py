"""Shared JSON output conversion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
from dataclasses import asdict, is_dataclass

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from src.state.contracts import JsonValue


def json_safe(value: object) -> JsonValue:
    """Convert dataclasses and paths into JSON-safe values."""
    result: JsonValue
    if not isinstance(value, type) and is_dataclass(value):
        dataclass_value: DataclassInstance = cast("DataclassInstance", value)
        result = json_safe(asdict(dataclass_value))
    elif isinstance(value, Path):
        result = str(value)
    elif isinstance(value, dict):
        items = cast("dict[object, object]", value)
        if any(not isinstance(key, str) for key in items):
            msg = "JSON object keys must be strings."
            raise TypeError(msg)
        result = {key: json_safe(item) for key, item in items.items() if isinstance(key, str)}
    elif isinstance(value, (list, tuple)):
        result = [json_safe(item) for item in cast("list[object] | tuple[object, ...]", value)]
    elif value is None or isinstance(value, str | int | float | bool):
        result = value
    else:
        msg = f"unsupported JSON value type: {type(value).__name__}"
        raise TypeError(msg)
    return result
