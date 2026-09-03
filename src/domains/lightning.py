"""Lightning-backed deterministic multi-task training loop."""

from __future__ import annotations

import math
import torch
import random
import numpy as np
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
from lightning.fabric.fabric import Fabric
from src.state.dashboard import TrainUpdate
from src.domains.labels import IGNORED_LABEL_ID
from src.state.loop import BatchLosses, TrainBatch

if TYPE_CHECKING:
    from src.state.loop import TrainRun
    from torch.utils.data import DataLoader
    from src.state.training import BatchData
    from collections.abc import Callable, Iterator


def setup_lightning_run(run: TrainRun) -> Fabric:
    """Attach the model and optimizer to one Fabric device."""
    fabric = Fabric(
        accelerator=run.run_device.type,
        devices=1,
        precision="16-mixed" if run.run_device.type == "cuda" else "32-true",
    )
    model, optimizer = fabric.setup(run.state.model, run.state.optimizer)
    run.state.model = cast("torch.nn.Module", model)
    run.state.optimizer = cast("torch.optim.Optimizer", optimizer)
    return fabric


def grad_accum_steps(micro_batch_size: int, override: int, batch_size: int) -> int:
    """Return a validated gradient accumulation step count."""
    if micro_batch_size < 1 or batch_size < 1 or override < 0:
        msg = "micro-batch size, batch size, and accumulation override must be valid positive values."
        raise TrainingError(msg)
    return override if override > 0 else max(1, math.ceil(batch_size / micro_batch_size))


def build_task_schedule(
    screen_batches: int,
    safety_batches: int,
    safety_probability: float,
    seed: int,
    epoch: int,
) -> list[bool]:
    """Build a deterministic schedule that covers both iterators at the requested ratio."""
    if screen_batches < 1 or safety_batches < 1:
        msg = "training iterators must both contain at least one batch."
        raise TrainingError(msg)
    if not 0 < safety_probability < 1:
        msg = "safety batch probability must be greater than zero and less than one."
        raise TrainingError(msg)
    total = max(
        math.ceil(screen_batches / (1 - safety_probability)),
        math.ceil(safety_batches / safety_probability),
    )
    safety_count = max(safety_batches, round(total * safety_probability))
    screen_count = max(screen_batches, total - safety_count)
    schedule = [False] * screen_count + [True] * safety_count
    generator = torch.Generator().manual_seed(seed + epoch)
    order: torch.Tensor = torch.randperm(len(schedule), generator=generator)
    return [schedule[int(order[position])] for position in range(len(schedule))]


def fit_lightning_epoch(
    run: TrainRun,
    fabric: Fabric,
    epoch: int,
    accumulation_steps: int,
) -> dict[str, float]:
    """Fit one deterministic epoch with correctly scaled accumulation groups."""
    run.state.model.train()
    schedule = build_task_schedule(
        len(run.iterators.train_screen),
        len(run.iterators.train_safety),
        run.args.safety_batch_probability,
        run.args.seed,
        epoch,
    )
    screen_batches = cycle_batches(run.iterators.train_screen)
    safety_batches = cycle_batches(run.iterators.train_safety)
    totals = torch.zeros(3, device=run.run_device, dtype=torch.float64)
    report_interval = max(1, len(schedule) // 100)

    run.dashboard.start_epoch(epoch, len(schedule), float(run.state.optimizer.param_groups[-1]["lr"]))
    for group_start in range(0, len(schedule), accumulation_steps):
        group = schedule[group_start : group_start + accumulation_steps]
        run.state.optimizer.zero_grad(set_to_none=True)

        for offset, is_safety_batch in enumerate(group):
            batch_index = group_start + offset
            batch = select_batch(
                screen_batches,
                safety_batches,
                is_safety_batch=is_safety_batch,
                run_device=run.run_device,
            )
            losses = train_batch(run, batch, is_safety_batch=is_safety_batch)
            fabric.backward(losses.loss / len(group))
            totals += torch.stack((losses.loss.detach(), losses.screen.detach(), losses.safety.detach())).to(
                dtype=torch.float64,
            )

            completed = batch_index + 1
            if completed % report_interval == 0 or completed == len(schedule):
                update_dashboard(run, completed, batch, totals)

        run.state.optimizer.step()

    run.state.optimizer.zero_grad(set_to_none=True)
    run.state.scheduler.step()
    total_loss, screen_loss, safety_loss = loss_values(totals)
    divisor = len(schedule)
    return {
        "loss": total_loss / divisor,
        "screen_loss": screen_loss / divisor,
        "safety_loss": safety_loss / divisor,
    }


def update_dashboard(run: TrainRun, completed: int, batch: TrainBatch, totals: torch.Tensor) -> None:
    """Synchronize accumulated losses for one dashboard update."""
    total_loss, screen_loss, safety_loss = loss_values(totals)
    run.dashboard.update_train(
        TrainUpdate(
            batch=completed,
            batch_images=batch.image_count,
            train_loss=total_loss / completed,
            screen_loss=screen_loss / completed,
            safety_loss=safety_loss / completed,
            learning_rate=float(run.state.optimizer.param_groups[-1]["lr"]),
        ),
    )


def loss_values(totals: torch.Tensor) -> tuple[float, float, float]:
    """Synchronize the three accumulated loss values once per report."""
    values = totals.detach().cpu()
    return float(values[0]), float(values[1]), float(values[2])


def cycle_batches(iterator: DataLoader[BatchData | None]) -> Iterator[BatchData]:
    """Cycle a nonempty iterator while rejecting epochs with no readable samples."""
    if len(iterator) == 0:
        msg = "training iterator is empty."
        raise TrainingError(msg)
    while True:
        yielded = False
        for batch in iterator:
            if batch is None:
                continue
            yielded = True
            yield batch
        if not yielded:
            msg = "training iterator produced no readable samples."
            raise TrainingError(msg)


def select_batch(
    screen_batches: Iterator[BatchData],
    safety_batches: Iterator[BatchData],
    *,
    is_safety_batch: bool,
    run_device: torch.device,
) -> TrainBatch:
    """Move the next scheduled batch to the run device."""
    images, screen_targets, safety_targets, _metas = next(safety_batches if is_safety_batch else screen_batches)
    return TrainBatch(
        images=images.to(run_device, non_blocking=True),
        screen_targets=screen_targets.to(run_device, non_blocking=True),
        safety_targets=safety_targets.to(run_device, non_blocking=True),
        image_count=int(images.shape[0]),
    )


def train_batch(run: TrainRun, batch: TrainBatch, *, is_safety_batch: bool) -> BatchLosses:
    """Compute both heads for screen data and only safety for safety-balanced data."""
    logits = run.state.model(batch.images)
    valid_count = (batch.safety_targets != IGNORED_LABEL_ID).sum().clamp_min(1)
    safety_loss = run.state.safety_criterion(logits["safety"], batch.safety_targets) / valid_count
    screen_loss = (
        logits["screen"].sum() * 0
        if is_safety_batch
        else run.state.screen_criterion(logits["screen"], batch.screen_targets)
    )
    return BatchLosses(
        loss=screen_loss + run.args.safety_loss_weight * safety_loss,
        screen=screen_loss,
        safety=safety_loss,
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    manual_seed: Callable[[int], torch.Generator] = cast("Callable[[int], torch.Generator]", torch.manual_seed)
    manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
