"""Validate datasets and materialize inputs for the fitting phase."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.domains.hub import cache_model
from src.domains.recipe.build import train_config
from src.config.model import SCREEN_FALLBACK_LABEL
from src.domains.artifacts.files import write_json
from src.domains.resume import validate_resume_config
from src.domains.samples.profile import profile_samples
from src.state.export import PreparedTraining, TrainJob
from src.domains.lightning import grad_accum_steps, set_seed
from src.domains.artifacts.paths import validated_dataset_path
from src.domains.images.preprocess import build_preprocess_spec
from src.domains.labels import IGNORED_LABEL_ID, build_label_state
from src.state.inputs import DatasetBuild, DatasetSpec, TrainConfig
from src.config.artifacts import TRAIN_LABELS_FILENAME, TRAIN_PROFILE_FILENAME
from src.state.training import (
    RawSplits,
    SampleIds,
    LabelState,
    SafetyCounts,
    SampleSplits,
    ScreenCounts,
    DatasetBundle,
)
from src.domains.artifacts.runs import (
    config_dir,
    train_config_path,
    skipped_images_path,
    prepare_local_artifact_dirs,
)
from src.domains.samples.dataset import (
    count_values,
    MultiTaskDataset,
    read_split_samples,
    remap_screen_samples,
    select_safety_labels,
    select_screen_labels,
    assert_split_coverage,
    filter_screen_samples,
)

if TYPE_CHECKING:
    from pathlib import Path
    from src.state.training import TrainArgs
    from src.state.metrics import TrainingProfile
    from src.state.dashboard import TrainDashboardProtocol

STAGE_CONFIG = "config"
STAGE_DATASET = "dataset"
STAGE_DATASETS = "datasets"
STAGE_FILTER = "filter"
STAGE_LABELS = "labels"
STAGE_MANIFEST = "manifest"
STAGE_MODEL_CACHE = "model cache"
STAGE_PROFILE = "profile"
STAGE_PROFILE_FILE = "profile file"
STAGE_SEED = "seed"


def prepare_training(job: TrainJob) -> PreparedTraining:
    """Prepare and validate every input needed by the fitting phase."""
    args = job.args
    dashboard = job.dashboard
    dataset_path = prepare_source(args, job.model_id, dashboard)
    preprocess = build_preprocess_spec(job.model_id)
    prepare_local_artifact_dirs(job.run_dir)
    raw_splits = read_manifest(dataset_path, dashboard)
    labels, fold_map = build_label_mapping(raw_splits, args, dashboard)
    samples = filter_splits(raw_splits, labels, fold_map, dashboard)
    counts = count_screen_splits(samples, labels)
    safety_counts = count_safety_splits(samples, labels)
    check_screen_counts(counts)
    check_safety_counts(safety_counts)
    profile = build_profile(samples, labels, dashboard)
    accumulation_steps = grad_accum_steps(args.micro_batch_size, args.grad_accum_steps, args.batch_size)
    ids = build_ids(samples, labels)
    config = TrainConfig(
        args=args,
        model_id=job.model_id,
        dataset_path=dataset_path,
        labels=labels,
        counts=counts,
        safety_counts=safety_counts,
        fold_map=fold_map,
        preprocess=preprocess,
        accumulation_steps=accumulation_steps,
    )
    validate_resume_config(job.saved_config, train_config(config))
    skipped_path = skipped_images_path(job.run_dir)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.write_text("", encoding="utf-8")
    write_profile(job.run_dir, profile, dashboard)
    write_train_config(config, job.run_dir, dashboard)
    datasets = build_datasets(
        DatasetBuild(
            samples=samples,
            labels=labels,
            args=args,
            preprocess=preprocess,
            skipped_path=skipped_path,
            dashboard=dashboard,
        ),
    )
    return PreparedTraining(labels=labels, profile=profile, ids=ids, datasets=datasets)


def prepare_source(args: TrainArgs, model_id: str, dashboard: TrainDashboardProtocol) -> Path:
    """Prepare source."""
    dashboard.set_stage(STAGE_SEED, 0, 1)
    set_seed(args.seed)
    dashboard.advance_stage()

    dashboard.set_stage(STAGE_DATASET, 0, 1)
    dataset_path = validated_dataset_path(args.dataset)
    dashboard.advance_stage()

    dashboard.set_stage(STAGE_MODEL_CACHE, 0, 1)
    cache_model(model_id)
    dashboard.advance_stage()
    return dataset_path


def read_manifest(dataset_path: Path, dashboard: TrainDashboardProtocol) -> RawSplits:
    """Read manifest."""
    dashboard.set_stage(STAGE_MANIFEST, 0, 3)
    train = read_split_samples(dataset_path, "train")
    dashboard.advance_stage()
    val = read_split_samples(dataset_path, "val")
    dashboard.advance_stage()
    test = read_split_samples(dataset_path, "test")
    dashboard.advance_stage()
    return RawSplits(train=train, val=val, test=test)


def build_label_mapping(
    raw_splits: RawSplits,
    args: TrainArgs,
    dashboard: TrainDashboardProtocol,
) -> tuple[LabelState, dict[str, str]]:
    """Build labels and fold map."""
    dashboard.set_stage(STAGE_LABELS, 0, 1)
    raw_screen = select_screen_labels(raw_splits.train, args.screen_labels)
    raw_counts = count_values((sample.screen_label for sample in raw_splits.train), raw_screen)
    validate_label_floor("screen", raw_counts, args.min_train_count)
    fold_map = screen_fold_mapping(raw_counts, args.min_train_count)
    screen = sorted({fold_map.get(label, label) for label in raw_screen})
    screen_train = filter_screen_samples(remap_screen_samples(raw_splits.train, fold_map), set(screen))
    safety = select_safety_labels(screen_train, args.min_train_count)
    labels = build_label_state(screen, safety)
    dashboard.advance_stage()
    return labels, fold_map


def validate_label_floor(name: str, counts: dict[str, int], min_count: int) -> None:
    """Validate label floor."""
    if any(count >= min_count for count in counts.values()):
        return
    msg = f"no {name} labels have at least {min_count} train samples."
    raise TrainingError(msg)


def screen_fold_mapping(raw_counts: dict[str, int], min_count: int) -> dict[str, str]:
    """Map sparse screen classes into the configured generic class."""
    return {
        label: SCREEN_FALLBACK_LABEL
        for label, count in raw_counts.items()
        if count < min_count and label != SCREEN_FALLBACK_LABEL
    }


def filter_splits(
    raw_splits: RawSplits,
    labels: LabelState,
    fold_map: dict[str, str],
    dashboard: TrainDashboardProtocol,
) -> SampleSplits:
    """Filter splits."""
    dashboard.set_stage(STAGE_FILTER, 0, 1)
    screen_label_set = set(labels.screen)
    train = filter_screen_samples(remap_screen_samples(raw_splits.train, fold_map), screen_label_set)
    val = filter_screen_samples(remap_screen_samples(raw_splits.val, fold_map), screen_label_set)
    test = filter_screen_samples(remap_screen_samples(raw_splits.test, fold_map), screen_label_set)
    if not train:
        msg = "no train samples are available after filtering to selected screen labels."
        raise TrainingError(msg)
    splits = SampleSplits(
        train=train,
        val=val,
        test=test,
    )
    dashboard.set_counts(
        len(labels.screen),
        len(labels.safety),
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )
    dashboard.advance_stage()
    return splits


def count_screen_splits(samples: SampleSplits, labels: LabelState) -> ScreenCounts:
    """Count every selected screen class across dataset splits."""
    return ScreenCounts(
        train=count_values((sample.screen_label for sample in samples.train), labels.screen),
        val=count_values((sample.screen_label for sample in samples.val), labels.screen),
        test=count_values((sample.screen_label for sample in samples.test), labels.screen),
    )


def count_safety_splits(samples: SampleSplits, labels: LabelState) -> SafetyCounts:
    """Count every selected safety class across dataset splits."""
    return SafetyCounts(
        train=count_values((sample.safety_label for sample in samples.train), labels.safety),
        val=count_values((sample.safety_label for sample in samples.val), labels.safety),
        test=count_values((sample.safety_label for sample in samples.test), labels.safety),
    )


def check_screen_counts(counts: ScreenCounts) -> None:
    """Check screen counts."""
    assert_split_coverage("train", counts.train)
    assert_split_coverage("val", counts.val)
    assert_split_coverage("test", counts.test)


def check_safety_counts(counts: SafetyCounts) -> None:
    """Check safety counts."""
    assert_split_coverage("train", counts.train)
    assert_split_coverage("val", counts.val)
    assert_split_coverage("test", counts.test)


def build_profile(
    samples: SampleSplits,
    labels: LabelState,
    dashboard: TrainDashboardProtocol,
) -> TrainingProfile:
    """Build profile."""
    dashboard.set_stage(STAGE_PROFILE, 0, 1)
    profile = profile_samples({"train": samples.train, "val": samples.val, "test": samples.test}, labels.screen)
    dashboard.advance_stage()
    return profile


def write_profile(run_dir: Path, profile: TrainingProfile, dashboard: TrainDashboardProtocol) -> None:
    """Write profile."""
    dashboard.set_stage(STAGE_PROFILE_FILE, 0, 1)
    write_json(config_dir(run_dir) / TRAIN_PROFILE_FILENAME, profile)
    dashboard.advance_stage()


def write_train_config(config: TrainConfig, run_dir: Path, dashboard: TrainDashboardProtocol) -> None:
    """Write train config."""
    dashboard.set_stage(STAGE_CONFIG, 0, 1)
    write_json(
        train_config_path(run_dir),
        train_config(config),
    )
    write_json(
        config_dir(run_dir) / TRAIN_LABELS_FILENAME,
        {
            "screen": config.labels.screen,
            "safety": config.labels.safety,
        },
    )
    dashboard.advance_stage()


def build_datasets(build: DatasetBuild) -> DatasetBundle:
    """Build datasets."""
    build.dashboard.set_stage(STAGE_DATASETS, 0, 3)
    train = MultiTaskDataset(
        DatasetSpec(
            build.samples.train,
            build.labels,
            build.args.image_size,
            build.preprocess,
            augment=True,
            skipped_path=build.skipped_path,
        ),
    )
    build.dashboard.advance_stage()
    val = MultiTaskDataset(
        DatasetSpec(
            build.samples.val,
            build.labels,
            build.args.image_size,
            build.preprocess,
            augment=False,
            skipped_path=build.skipped_path,
        ),
    )
    build.dashboard.advance_stage()
    test = MultiTaskDataset(
        DatasetSpec(
            build.samples.test,
            build.labels,
            build.args.image_size,
            build.preprocess,
            augment=False,
            skipped_path=build.skipped_path,
        ),
    )
    build.dashboard.advance_stage()
    return DatasetBundle(train=train, val=val, test=test)


def build_ids(samples: SampleSplits, labels: LabelState) -> SampleIds:
    """Build ids."""
    return SampleIds(
        train_screen=[labels.screen_to_id[sample.screen_label] for sample in samples.train],
        train_safety=[labels.safety_to_id.get(sample.safety_label, IGNORED_LABEL_ID) for sample in samples.train],
        val_screen=[labels.screen_to_id[sample.screen_label] for sample in samples.val],
        test_screen=[labels.screen_to_id[sample.screen_label] for sample in samples.test],
        val_safety=[labels.safety_to_id.get(sample.safety_label, IGNORED_LABEL_ID) for sample in samples.val],
        test_safety=[labels.safety_to_id.get(sample.safety_label, IGNORED_LABEL_ID) for sample in samples.test],
    )
