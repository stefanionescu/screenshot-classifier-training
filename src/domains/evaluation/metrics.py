"""Evaluate torch and ONNX models with accuracy and latency metrics."""

from __future__ import annotations

import time
import torch
import numpy as np
from typing import TYPE_CHECKING, cast
from src.domains.labels import IGNORED_LABEL_ID
from src.domains.model.hardware import cpu_label
from src.domains.model.network import public_logits
from src.domains.model.onnx.runtime import create_onnx_session
from src.state.metrics import EvaluationMetrics, TimingContext, TimingMetrics
from src.domains.evaluation.stats import latency_stats, metrics_from_confusion
from src.config.model import TRAIN_DEVICE_CPU, TRAIN_DEVICE_CUDA, TRAIN_DEVICE_MPS
from src.domains.evaluation.diagnostics import prediction_rows, write_failures_from_rows, write_prediction_rows
from src.state.eval import (
    OnnxEval,
    OnnxLoop,
    PredBatch,
    TorchEval,
    TimedBatch,
    MetricCounts,
    EvalArtifacts,
    LatencySamples,
)
from src.config.artifacts import (
    ONNX_INPUT_NAME,
    ONNX_CPU_PROVIDERS,
    ONNX_TIMING_PROVIDER,
    ONNX_SAFETY_OUTPUT_NAME,
    ONNX_SCREEN_OUTPUT_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from torch.utils.data import DataLoader
    from src.state.training import BatchData
    from src.state.dashboard import TrainDashboardProtocol

DEFAULT_TEST_STAGE = "test full"


def evaluate(run: TorchEval) -> EvaluationMetrics:
    """Evaluate a torch model."""
    run.model.eval()
    counts = metric_counts(run.labels.screen, run.labels.safety)
    latency = LatencySamples()
    if run.dashboard is not None:
        run.dashboard.start_eval(run.description, len(run.iterator))
    with torch.no_grad():
        for batch in eval_batches(run.iterator):
            record_torch_batch(run, counts, latency, batch)
    return eval_result(counts, run.labels.screen, run.labels.safety, run.timing, latency)


def wait_device(run_device: torch.device) -> None:
    """Wait for queued accelerator work before recording latency."""
    if run_device.type == TRAIN_DEVICE_CUDA:
        torch.cuda.synchronize(run_device)
    if run_device.type == TRAIN_DEVICE_MPS:
        torch.mps.synchronize()


def add_latency(samples: list[float], seconds: float, count: int) -> None:
    """Add one honest batch-normalized observation."""
    if count <= 0:
        return
    samples.append(seconds / count)


def eval_batches(iterator: DataLoader[BatchData | None]) -> Iterator[TimedBatch]:
    """Yield batches with one-based indexes and measured iterator latency."""
    batch_iterator = iter(iterator)
    for batch_index in range(1, len(iterator) + 1):
        try:
            read_started = time.perf_counter()
            batch = next(batch_iterator)
            yield TimedBatch(batch_index, batch, time.perf_counter() - read_started)
        except StopIteration:
            break


def metric_counts(screen_labels: list[str], safety_labels: list[str]) -> MetricCounts:
    """Create zeroed metric counters."""
    return MetricCounts(
        screen_confusion=torch.zeros((len(screen_labels), len(screen_labels)), dtype=torch.long),
        safety_confusion=torch.zeros((len(safety_labels), len(safety_labels)), dtype=torch.long),
    )


def update_eval_dashboard(dashboard: TrainDashboardProtocol | None, batch_index: int) -> None:
    """Update eval progress when a dashboard is active."""
    if dashboard is not None:
        dashboard.eval_done = batch_index
        dashboard.refresh()


def write_pred_rows(artifacts: EvalArtifacts | None, batch: PredBatch) -> None:
    """Write prediction rows when artifacts are enabled."""
    if artifacts is None or not artifacts.write_predictions:
        return
    rows = prediction_rows(artifacts, batch)
    write_prediction_rows(artifacts, rows)
    write_failures_from_rows(artifacts.run_dir, rows)


def record_torch_batch(
    run: TorchEval,
    counts: MetricCounts,
    latency: LatencySamples,
    batch: TimedBatch,
) -> None:
    """Record metrics for one torch batch."""
    if batch.batch is None:
        update_eval_dashboard(run.dashboard, batch.index)
        return
    processing_start = time.perf_counter()
    images, screen_targets, safety_targets, metas = batch.batch
    batch_images = int(images.shape[0])
    images = images.to(run.run_device, non_blocking=True)
    screen_targets = screen_targets.to(run.run_device, non_blocking=True)
    safety_targets = safety_targets.to(run.run_device, non_blocking=True)
    wait_device(run.run_device)
    model_start = time.perf_counter()
    logits = run.model(images)
    wait_device(run.run_device)
    model_stop = time.perf_counter()
    public = public_logits(logits)
    screen_logits = public[ONNX_SCREEN_OUTPUT_NAME]
    safety_logits = public[ONNX_SAFETY_OUTPUT_NAME]
    pred_batch = PredBatch(
        metas=metas,
        screen_labels=run.labels.screen,
        safety_labels=run.labels.safety,
        screen_logits=screen_logits,
        safety_logits=safety_logits,
        screen_targets=screen_targets,
        safety_targets=safety_targets,
    )
    record_torch_predictions(counts, pred_batch)
    write_pred_rows(
        run.artifacts,
        pred_batch,
    )
    if run.timing is not None:
        total_seconds = batch.read_seconds + time.perf_counter() - processing_start
        add_latency(latency.read, batch.read_seconds, batch_images)
        add_latency(latency.model, model_stop - model_start, batch_images)
        add_latency(latency.total, total_seconds, batch_images)
        latency.image_count += batch_images
        latency.elapsed += total_seconds
    update_eval_dashboard(run.dashboard, batch.index)


def record_torch_predictions(
    counts: MetricCounts,
    batch: PredBatch,
) -> None:
    """Record torch logits and targets."""
    screen_logits = cast("torch.Tensor", batch.screen_logits)
    safety_logits = cast("torch.Tensor", batch.safety_logits)
    screen_targets = cast("torch.Tensor", batch.screen_targets)
    safety_targets = cast("torch.Tensor", batch.safety_targets)
    topk = screen_logits.softmax(dim=1).topk(k=min(2, len(batch.screen_labels)), dim=1)
    counts.screen_top2_hits += int((topk.indices == screen_targets.unsqueeze(1)).any(dim=1).sum().cpu())
    counts.safety_top2_hits += record_safety_top2(safety_logits, safety_targets, len(batch.safety_labels))
    for target, prediction in zip(screen_targets.cpu(), screen_logits.argmax(dim=1).cpu(), strict=True):
        counts.screen_confusion[int(target), int(prediction)] += 1
        counts.screen_total += 1
    for target, prediction in zip(safety_targets.cpu(), safety_logits.argmax(dim=1).cpu(), strict=True):
        target_id = int(target)
        if target_id != IGNORED_LABEL_ID:
            counts.safety_confusion[target_id, int(prediction)] += 1
            counts.safety_total += 1


def eval_result(
    counts: MetricCounts,
    screen_labels: list[str],
    safety_labels: list[str],
    timing: TimingContext | None,
    latency: LatencySamples,
) -> EvaluationMetrics:
    """Build an evaluation result."""
    metrics = EvaluationMetrics(
        screen=metrics_from_confusion(
            counts.screen_confusion,
            screen_labels,
            counts.screen_top2_hits,
            counts.screen_total,
        ),
        safety=metrics_from_confusion(
            counts.safety_confusion,
            safety_labels,
            counts.safety_top2_hits,
            counts.safety_total,
        ),
    )
    if timing is not None:
        metrics["timing"] = timing_report(timing, latency)
    return metrics


def timing_report(
    timing: TimingContext,
    latency: LatencySamples,
) -> TimingMetrics:
    """Build timing metadata from batch-normalized measurements."""
    return TimingMetrics(
        format=timing["format"],
        provider=timing["provider"],
        device=timing["device"],
        hardware=timing["hardware"],
        image_size=timing["image_size"],
        batch_size=timing["batch_size"],
        image_count=latency.image_count,
        images_per_second=latency.image_count / latency.elapsed if latency.elapsed > 0 else 0.0,
        read_preprocess=latency_stats(latency.read),
        model_run=latency_stats(latency.model),
        total=latency_stats(latency.total),
    )


def evaluate_onnx(run: OnnxEval) -> EvaluationMetrics:
    """Evaluate onnx."""
    counts = metric_counts(run.screen_labels, run.safety_labels)
    latency = LatencySamples()
    loop = OnnxLoop(
        run=run,
        session=create_onnx_session(run.model_path, ONNX_CPU_PROVIDERS),
    )
    if run.dashboard is not None:
        run.dashboard.start_eval(run.stage or DEFAULT_TEST_STAGE, len(run.iterator))
    for batch in eval_batches(run.iterator):
        record_onnx_batch(loop, counts, latency, batch)
    return eval_result(counts, run.screen_labels, run.safety_labels, onnx_timing(run), latency)


def onnx_timing(run: OnnxEval) -> TimingContext:
    """Build ONNX timing metadata."""
    return TimingContext(
        format=run.model_format,
        provider=ONNX_TIMING_PROVIDER,
        device=TRAIN_DEVICE_CPU,
        hardware=cpu_label(),
        image_size=run.image_size,
        batch_size=run.batch_size,
    )


def record_onnx_batch(
    loop: OnnxLoop,
    counts: MetricCounts,
    latency: LatencySamples,
    batch: TimedBatch,
) -> None:
    """Record metrics for one ONNX batch."""
    if batch.batch is None:
        update_eval_dashboard(loop.run.dashboard, batch.index)
        return
    processing_start = time.perf_counter()
    images, screen_targets, safety_targets, metas = batch.batch
    batch_images = int(images.shape[0])
    model_start = time.perf_counter()
    outputs = loop.session.run(None, {ONNX_INPUT_NAME: images.contiguous().numpy()})
    screen_logits = outputs[0]
    safety_logits = outputs[1]
    model_stop = time.perf_counter()
    write_pred_rows(
        loop.run.artifacts,
        PredBatch(
            metas=metas,
            screen_labels=loop.run.screen_labels,
            safety_labels=loop.run.safety_labels,
            screen_logits=screen_logits,
            safety_logits=safety_logits,
            screen_targets=screen_targets.numpy(),
            safety_targets=safety_targets.numpy(),
        ),
    )
    counts.screen_top2_hits += record_screen_batch(counts.screen_confusion, screen_logits, screen_targets.numpy())
    safety_hits, safety_total = record_safety_batch(counts.safety_confusion, safety_logits, safety_targets.numpy())
    counts.safety_top2_hits += safety_hits
    counts.safety_total += safety_total
    counts.screen_total += batch_images
    total_seconds = batch.read_seconds + time.perf_counter() - processing_start
    add_latency(latency.read, batch.read_seconds, batch_images)
    add_latency(latency.model, model_stop - model_start, batch_images)
    add_latency(latency.total, total_seconds, batch_images)
    latency.image_count += batch_images
    latency.elapsed += total_seconds
    update_eval_dashboard(loop.run.dashboard, batch.index)


def record_screen_batch(confusion: torch.Tensor, logits: np.ndarray, targets: np.ndarray) -> int:
    """Record screen batch."""
    predictions = logits.argmax(axis=1)
    top_count = min(2, logits.shape[1])
    top_ids = np.argsort(logits, axis=1)[:, -top_count:]
    top_hits = int((top_ids == targets.reshape(-1, 1)).any(axis=1).sum())
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[int(target), int(prediction)] += 1
    return top_hits


def record_safety_top2(logits: torch.Tensor, targets: torch.Tensor, label_count: int) -> int:
    """Record torch safety top2."""
    valid = targets != IGNORED_LABEL_ID
    if not bool(valid.any()):
        return 0
    topk = logits.softmax(dim=1).topk(k=min(2, label_count), dim=1)
    return int((topk.indices[valid] == targets[valid].unsqueeze(1)).any(dim=1).sum().cpu())


def record_safety_batch(
    confusion: torch.Tensor,
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[int, int]:
    """Record safety batch."""
    predictions = logits.argmax(axis=1)
    top_count = min(2, logits.shape[1])
    top_ids = np.argsort(logits, axis=1)[:, -top_count:]
    top_hits = 0
    total = 0
    for target, prediction, top_row in zip(targets, predictions, top_ids, strict=True):
        target_id = int(target)
        if target_id == IGNORED_LABEL_ID:
            continue
        prediction_id = int(prediction)
        confusion[target_id, prediction_id] += 1
        if target_id in top_row:
            top_hits += 1
        total += 1
    return top_hits, total
