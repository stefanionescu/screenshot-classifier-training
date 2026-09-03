"""Parse training options and invoke the training application boundary."""

from __future__ import annotations

import math
import argparse
from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.domains.train.run import run_train
from src.state.train.training import TrainArgs
from src.cli.lib.boundary import cli_error_boundary
from src.cli.lib.env import configure_cli_environment
from src.config.train.artifacts import ONNX_MIN_DIMENSION
from src.config.train.cli import (
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_EPOCHS,
    DEFAULT_WORKERS,
    EVAL_CLASS_LIMIT,
    TRAIN_CLI_PROGRAM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WEIGHT_DECAY,
    TRAIN_CLI_DESCRIPTION,
    DEFAULT_GRAD_ACCUM_STEPS,
)
from src.config.train.model import (
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    SAFETY_MAX_REPEAT,
    SCREEN_MAX_REPEAT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    SAFETY_LOSS_WEIGHT,
    SAFETY_TARGET_RATIO,
    SCREEN_TARGET_RATIO,
    DEFAULT_MIN_TRAIN_COUNT,
    SAFETY_BATCH_PROBABILITY,
    TARGET_EFFECTIVE_BATCH_SIZE,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> TrainArgs:
    """Parse and validate training arguments."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    args = TrainArgs(**vars(namespace))
    _fail_invalid_args(
        (
            (args.push and args.push_only, "--push and --push-only cannot be used together."),
            (args.image_size < ONNX_MIN_DIMENSION, f"--size must be at least {ONNX_MIN_DIMENSION}."),
            (args.epochs < 1, "--epochs must be a positive integer."),
            (args.micro_batch_size < 1, "--micro-batch must be a positive integer."),
            (args.workers < 0, "--workers must be zero or greater."),
            (args.grad_accum_steps < 0, "--accumulation must be zero or greater."),
            (args.batch_size < 1, "--batch must be a positive integer."),
            (args.eval_class_limit < 1, "--eval-limit must be a positive integer."),
            (args.min_train_count < 1, "--min-count must be a positive integer."),
            (not math.isfinite(args.lr) or args.lr <= 0, "--lr must be a finite positive number."),
            (
                not math.isfinite(args.weight_decay) or args.weight_decay < 0,
                "--decay must be a finite non-negative number.",
            ),
            (
                not math.isfinite(args.screen_target_ratio)
                or not math.isfinite(args.safety_target_ratio)
                or args.screen_target_ratio < 1
                or args.safety_target_ratio < 1,
                "Target effective ratios must be finite numbers of at least 1.",
            ),
            (args.screen_max_repeat < 1 or args.safety_max_repeat < 1, "Repeat caps must be positive integers."),
            (
                not math.isfinite(args.safety_loss_weight) or args.safety_loss_weight < 0,
                "--safety-weight must be finite and non-negative.",
            ),
            (
                not math.isfinite(args.safety_batch_probability) or not 0 < args.safety_batch_probability < 1,
                "--safety-share must be finite, greater than zero, and less than one.",
            ),
        ),
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(
        prog=TRAIN_CLI_PROGRAM,
        description=TRAIN_CLI_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Local dataset directory or Hub dataset ID.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Supported timm Hugging Face backbone ID.")
    parser.add_argument(
        "--screens",
        dest="screen_labels",
        nargs="*",
        default=None,
        help="Screen labels to train; omit the option or provide no values to select every available label.",
    )
    parser.add_argument(
        "--size",
        dest="image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Longest input image side.",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Run name below the model output root.",
    )
    parser.add_argument("--repo", default=None, help="Destination model repository ID.")
    parser.add_argument("--push", action="store_true", help="Publish the completed export.")
    parser.add_argument("--push-only", action="store_true", help="Publish an existing validated export.")
    parser.add_argument("--public", action="store_true", help="Create or update a public Hub repository.")
    _add_optimization_arguments(parser)
    _add_sampling_arguments(parser)
    _add_export_arguments(parser)
    return parser


def _add_optimization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add batch, optimizer, schedule, and reproducibility controls."""
    group = parser.add_argument_group("Optimization")
    group.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Total epoch target.")
    group.add_argument(
        "--micro-batch",
        dest="micro_batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Images processed per gradient-accumulation step.",
    )
    group.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Data-iterator worker processes.")
    group.add_argument("--lr", type=float, default=DEFAULT_LR, help="Task-head learning rate.")
    group.add_argument(
        "--decay",
        dest="weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay.",
    )
    group.add_argument(
        "--accumulation",
        dest="grad_accum_steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
        help="Explicit accumulation steps; zero derives the value from --batch.",
    )
    group.add_argument(
        "--batch",
        dest="batch_size",
        type=int,
        default=TARGET_EFFECTIVE_BATCH_SIZE,
        help="Target batch size across gradient-accumulation steps.",
    )
    group.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic training seed.")
    group.add_argument(
        "--eval-limit",
        dest="eval_class_limit",
        type=int,
        default=EVAL_CLASS_LIMIT,
        help="Maximum samples per class in balanced evaluation subsets.",
    )
    group.add_argument(
        "--min-count",
        dest="min_train_count",
        type=int,
        default=DEFAULT_MIN_TRAIN_COUNT,
        help="Minimum label count.",
    )


def _add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    """Add class balancing and multitask weighting controls."""
    group = parser.add_argument_group("Sampling and task weighting")
    group.add_argument(
        "--screen-ratio",
        dest="screen_target_ratio",
        type=float,
        default=SCREEN_TARGET_RATIO,
        help="Screen balance ratio.",
    )
    group.add_argument(
        "--safety-ratio",
        dest="safety_target_ratio",
        type=float,
        default=SAFETY_TARGET_RATIO,
        help="Safety balance ratio.",
    )
    group.add_argument(
        "--screen-repeat",
        dest="screen_max_repeat",
        type=int,
        default=SCREEN_MAX_REPEAT,
        help="Screen repeat cap.",
    )
    group.add_argument(
        "--safety-repeat",
        dest="safety_max_repeat",
        type=int,
        default=SAFETY_MAX_REPEAT,
        help="Safety repeat cap.",
    )
    group.add_argument(
        "--safety-weight",
        dest="safety_loss_weight",
        type=float,
        default=SAFETY_LOSS_WEIGHT,
        help="Safety loss multiplier.",
    )
    group.add_argument(
        "--safety-share",
        dest="safety_batch_probability",
        type=float,
        default=SAFETY_BATCH_PROBABILITY,
        help="Target share of dedicated safety batches in each epoch schedule.",
    )


def _add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Add resume and model-export controls."""
    group = parser.add_argument_group("Resume and export")
    group.add_argument(
        "--export",
        choices=("fp16", "fp32"),
        default="fp16",
        help="ONNX export format.",
    )
    group.add_argument("--resume", action="store_true", help="Resume the latest run.")
    group.add_argument(
        "--checkpoint",
        dest="export_checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Checkpoint used for the final model export.",
    )


def _fail_invalid_args(checks: tuple[tuple[bool, str], ...]) -> None:
    """Raise the first matching validation error."""
    for invalid, message in checks:
        if invalid:
            raise TrainingError(message)


def main() -> None:
    """Run the command."""
    configure_cli_environment()
    with cli_error_boundary():
        run_train(parse_args())


if __name__ == "__main__":
    main()
