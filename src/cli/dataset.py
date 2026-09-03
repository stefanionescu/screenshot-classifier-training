"""Dataset build command entrypoint."""

from __future__ import annotations

import sys
import math
import asyncio
import argparse
from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.cli.errors import report_errors
from src.config.dataset import DATASET_CONFIG
from src.cli.env import configure_cli_environment
from src.domains.dataset.build import build_dataset
from src.shared.runtime.progress import ProgressBar
from src.shared.console import write_lines, write_command_fields
from src.state.dataset import DatasetBuildOptions, DatasetCommandArgs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from src.state.dataset import DatasetProgressEvent


def _build_parser() -> argparse.ArgumentParser:
    """Build the dataset command parser."""
    parser = argparse.ArgumentParser(
        prog="dataset",
        description="Build the phone screenshot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=DATASET_CONFIG["output"]["default_dir"],
        help="Output directory below the project output root.",
    )
    parser.add_argument(
        "--validation",
        dest="val_percent",
        type=float,
        default=DATASET_CONFIG["split"]["default_val_percent"],
        help="Validation percentage allocated within each screen-and-safety bucket.",
    )
    parser.add_argument(
        "--test",
        dest="test_percent",
        type=float,
        default=DATASET_CONFIG["split"]["default_test_percent"],
        help="Test percentage allocated within each screen-and-safety bucket.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> DatasetCommandArgs:
    """Parse dataset command arguments."""
    namespace = _build_parser().parse_args(argv)
    args = DatasetCommandArgs(
        output_dir=namespace.output_dir,
        val_percent=namespace.val_percent,
        test_percent=namespace.test_percent,
    )
    if not math.isfinite(args.val_percent) or not math.isfinite(args.test_percent):
        msg = "split percentages must be finite numbers."
        raise TrainingError(msg)
    if args.val_percent < 0 or args.test_percent < 0:
        msg = "split percentages must not be negative."
        raise TrainingError(msg)
    if args.val_percent + args.test_percent > DATASET_CONFIG["split"]["max_held_out_percent"]:
        msg = "--validation plus --test must be 50 or less."
        raise TrainingError(msg)
    return args


def _dataset_progress() -> tuple[Callable[[DatasetProgressEvent], None], Callable[[], None]]:
    """Create progress callbacks for dataset build phases."""
    labels = {
        "entries": "Preparing dataset entries",
        "shards": "Writing data shards",
        "manifest": "Writing dataset manifests",
    }
    phase: str | None = None
    progress: ProgressBar | None = None

    def stop() -> None:
        """Stop the active dataset phase progress bar."""
        nonlocal phase, progress
        if progress is not None:
            progress.stop()
        phase = None
        progress = None

    def update(event: DatasetProgressEvent) -> None:
        """Update dataset phase progress from a build event."""
        nonlocal phase, progress
        if event.phase != phase:
            stop()
            phase = event.phase
            write_lines((f"\n  {labels[event.phase]}:", ""))
            progress = ProgressBar(event.total)
        if progress is None:
            msg = "dataset progress phase was not initialized."
            raise RuntimeError(msg)
        progress.update(event.completed)
        if event.completed >= event.total:
            stop()

    return update, stop


async def _run(argv: list[str]) -> None:
    """Run the dataset build from CLI arguments."""
    args = parse_args(argv)
    write_command_fields(
        None,
        (
            ("task", "Dataset build"),
            ("output", args.output_dir),
            ("validation", f"{args.val_percent:g}%"),
            ("test", f"{args.test_percent:g}%"),
        ),
    )
    update, stop = _dataset_progress()
    try:
        result = await build_dataset(
            DatasetBuildOptions(
                output_dir=args.output_dir,
                val_percent=args.val_percent,
                test_percent=args.test_percent,
            ),
            update,
        )
    finally:
        stop()
    write_command_fields(
        "Dataset complete",
        (
            ("output", result.output_dir),
            ("images", str(result.summary["total_images"])),
            ("rejected", str(result.summary["rejected_images"])),
            *((f"{split} shards", str(count)) for split, count in result.summary["shard_counts"].items()),
        ),
    )


def main() -> None:
    """Run the dataset command."""
    configure_cli_environment()
    with report_errors():
        asyncio.run(_run(sys.argv[1:]))


if __name__ == "__main__":
    main()
