"""Concurrent task execution utilities."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

T = TypeVar("T")
R = TypeVar("R")


async def _worker[T, R](
    pending: asyncio.Queue[tuple[int, T]],
    completed: asyncio.Queue[tuple[int, T, R]],
    run_item: Callable[[T, int], Awaitable[R]],
) -> None:
    """Drain work from one bounded worker slot."""
    while True:
        try:
            index, item = pending.get_nowait()
        except asyncio.QueueEmpty:
            return
        result = await run_item(item, index)
        await completed.put((index, item, result))


async def _consume[T, R](
    completed_results: asyncio.Queue[tuple[int, T, R]],
    results: dict[int, R],
    item_count: int,
    on_progress: Callable[[int, int, T], None] | None,
    on_result: Callable[[R, int, T], Awaitable[None]] | None,
) -> None:
    """Serialize result storage, durable callbacks, and progress updates."""
    for completed in range(1, item_count + 1):
        index, item, result = await completed_results.get()
        results[index] = result
        if on_result is not None:
            await on_result(result, index, item)
        if on_progress is not None:
            on_progress(completed, item_count, item)


async def run_concurrent[T, R](
    items: Sequence[T],
    run_item: Callable[[T, int], Awaitable[R]],
    concurrency: int,
    on_progress: Callable[[int, int, T], None] | None = None,
    on_result: Callable[[R, int, T], Awaitable[None]] | None = None,
) -> list[R]:
    """Run async work with bounded concurrency and ordered results."""
    if concurrency < 1:
        msg = "concurrency must be a positive integer."
        raise ValueError(msg)
    if not items:
        return []

    pending: asyncio.Queue[tuple[int, T]] = asyncio.Queue()
    completed_results: asyncio.Queue[tuple[int, T, R]] = asyncio.Queue()
    for index, item in enumerate(items):
        pending.put_nowait((index, item))
    results: dict[int, R] = {}

    async with asyncio.TaskGroup() as tasks:
        for _ in range(min(concurrency, len(items))):
            tasks.create_task(_worker(pending, completed_results, run_item))
        tasks.create_task(_consume(completed_results, results, len(items), on_progress, on_result))
    return [results[index] for index in range(len(items))]
