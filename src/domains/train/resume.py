"""Resume training recipe checks."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.config.train.cli import TRAIN_RESUME_OVERRIDES
from src.domains.train.artifacts.files import read_json
from src.domains.train.recipe.schema import parse_training_recipe
from src.domains.train.artifacts.run_files import train_config_path

if TYPE_CHECKING:
    from pathlib import Path
    from src.state.train.training import TrainArgs
    from src.state.train.recipe import TrainingRecipe


def resume_args(args: TrainArgs, saved_config: TrainingRecipe | None) -> TrainArgs:
    """Return arguments aligned with a validated saved recipe."""
    if not args.resume:
        return args
    if saved_config is None:
        msg = "--resume needs an existing training recipe."
        raise TrainingError(msg)
    labels = saved_config["labels"]["screen"]
    if args.screen_labels is not None and args.screen_labels != labels:
        msg = "--resume screen labels must match train.json labels."
        raise TrainingError(msg)
    selection = list(labels)
    for label in saved_config["sampling"]["folded_screen_labels"]:
        if label not in selection:
            selection.append(label)
    return replace(args, screen_labels=selection)


def read_train_config(run_dir: Path) -> TrainingRecipe | None:
    """Read and validate a saved training recipe when present."""
    path = train_config_path(run_dir)
    return parse_training_recipe(read_json(path)) if path.is_file() else None


def validate_resume_config(saved_config: TrainingRecipe | None, current_config: TrainingRecipe) -> None:
    """Validate current arguments against a saved training recipe."""
    if saved_config is None:
        return
    if saved_config["training"]["epochs"] != current_config["training"]["epochs"]:
        msg = "--resume cannot change --epochs because the saved cosine learning-rate schedule is fixed."
        raise TrainingError(msg)
    comparable = copy.deepcopy(current_config)
    if "workers" in TRAIN_RESUME_OVERRIDES:
        comparable["training"]["workers"] = saved_config["training"]["workers"]
    if comparable != saved_config:
        msg = "--resume arguments must match the saved training recipe."
        raise TrainingError(msg)
