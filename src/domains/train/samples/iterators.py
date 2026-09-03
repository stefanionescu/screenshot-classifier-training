"""Assemble adaptive training and balanced evaluation data iterators."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.state.train.inputs import IteratorConfig
from src.state.train.training import BatchData, BatchIterators, DatasetBundle, MetricIterators, SampleIds
from src.domains.train.samples.sampling import (
    build_eval_iterator,
    build_balanced_subset,
    build_adaptive_train_iterator,
)

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from src.state.train.training import TrainArgs
    from src.state.train.dashboard import TrainDashboardProtocol

ITERATORS_STAGE = "iterators"


def build_batch_iterators(
    datasets: DatasetBundle,
    ids: SampleIds,
    args: TrainArgs,
    run_device: torch.device,
    dashboard: TrainDashboardProtocol,
) -> BatchIterators:
    """Build batch iterators."""
    dashboard.set_stage(ITERATORS_STAGE, 0, 4)
    screen_iterator, screen_sampler = build_adaptive_train_iterator(
        IteratorConfig(
            dataset=datasets.train,
            labels=ids.train_screen,
            target_ratio=args.screen_target_ratio,
            max_repeat=args.screen_max_repeat,
            seed=args.seed,
            batch_size=args.micro_batch_size,
            workers=args.workers,
            run_device=run_device,
        ),
    )
    dashboard.advance_stage()
    safety_iterator, safety_sampler = build_adaptive_train_iterator(
        IteratorConfig(
            dataset=datasets.train,
            labels=ids.train_safety,
            target_ratio=args.safety_target_ratio,
            max_repeat=args.safety_max_repeat,
            seed=args.seed + 1,
            batch_size=args.micro_batch_size,
            workers=args.workers,
            run_device=run_device,
        ),
    )
    dashboard.advance_stage()
    raw_iterators = build_split_iterators(datasets, args, run_device)
    dashboard.advance_stage()
    metric_iterators = build_metric_iterators(datasets, ids, args, run_device)
    dashboard.advance_stage()
    return BatchIterators(
        train_screen=screen_iterator,
        train_safety=safety_iterator,
        screen_sampler=screen_sampler,
        safety_sampler=safety_sampler,
        ids=ids,
        val=raw_iterators[0],
        test=raw_iterators[1],
        val_screen=metric_iterators.val_screen,
        test_screen=metric_iterators.test_screen,
        val_safety=metric_iterators.val_safety,
        test_safety=metric_iterators.test_safety,
    )


def build_split_iterators(
    datasets: DatasetBundle,
    args: TrainArgs,
    run_device: torch.device,
) -> tuple[DataLoader[BatchData | None], DataLoader[BatchData | None]]:
    """Build split iterators."""
    return (
        build_eval_iterator(datasets.val, args.micro_batch_size, args.workers, run_device),
        build_eval_iterator(datasets.test, args.micro_batch_size, args.workers, run_device),
    )


def build_metric_iterators(
    datasets: DatasetBundle,
    ids: SampleIds,
    args: TrainArgs,
    run_device: torch.device,
) -> MetricIterators:
    """Build metric iterators."""
    return MetricIterators(
        val_screen=build_eval_iterator(
            build_balanced_subset(datasets.val, ids.val_screen, args.eval_class_limit, args.seed),
            args.micro_batch_size,
            args.workers,
            run_device,
        ),
        test_screen=build_eval_iterator(
            build_balanced_subset(datasets.test, ids.test_screen, args.eval_class_limit, args.seed),
            args.micro_batch_size,
            args.workers,
            run_device,
        ),
        val_safety=build_eval_iterator(
            build_balanced_subset(datasets.val, ids.val_safety, args.eval_class_limit, args.seed),
            args.micro_batch_size,
            args.workers,
            run_device,
        ),
        test_safety=build_eval_iterator(
            build_balanced_subset(datasets.test, ids.test_safety, args.eval_class_limit, args.seed),
            args.micro_batch_size,
            args.workers,
            run_device,
        ),
    )
