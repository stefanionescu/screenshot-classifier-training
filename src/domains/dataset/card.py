"""Build the Hugging Face dataset card."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import UTC, datetime
from src.config.image import IMAGE_CONFIG
from src.config.dataset import DATASET_CONFIG

if TYPE_CHECKING:
    from src.state.dataset import DatasetBuildOptions, DatasetSummary

TEMPLATE_PATH = Path(__file__).with_name("template.md")
PLACEHOLDER_PATTERN = re.compile(r"{{([a-z_]+)}}")
SIZE_CATEGORIES = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
)


def _split_replacements(summary: DatasetSummary, options: DatasetBuildOptions) -> dict[str, str]:
    """Build split, shard, artifact, and usage replacement values."""
    output = DATASET_CONFIG["output"]
    split_names = DATASET_CONFIG["split"]["names"]
    shard_counts = summary["shard_counts"]
    split_percentages = {
        "train": 100 - options.val_percent - options.test_percent,
        "val": options.val_percent,
        "test": options.test_percent,
    }
    split_shards = "\n".join(
        line
        for split in split_names
        for line in (
            f"          - split: {split}",
            f"            path: '{output['shards_dir']}/{split}/*{output['shard_extension']}'",
        )
    )
    split_stats = "\n".join(
        f"| `{split}` | {shard_counts[split]:,} | {split_percentages[split]:g}% |" for split in split_names
    )
    split_files = "\n".join(
        f"- `{output['shards_dir']}/{split}/*{output['shard_extension']}`: `{split}` samples." for split in split_names
    )
    artifact_files = "\n".join(
        (
            split_files,
            f"- `{output['metadata_file']}`: image, shard, screen, and safety counts.",
            f"- `{output['manifest_jsonl_file']}`: one JSON line per sample.",
            (
                f"- `{output['manifest_parquet_dir']}/split=*/"
                f"{output['manifest_parquet_part_file']}`: Parquet rows grouped by split."
            ),
            f"- `{output['rejected_jsonl_file']}`: sources excluded from the build.",
            f"- `{output['readme_file']}`: this generated dataset card.",
        ),
    )
    usage_examples = "\n".join(
        f'{split} = load_dataset("{DATASET_CONFIG["repo"]["id"]}", split="{split}")' for split in split_names
    )
    return {
        "artifact_files": artifact_files,
        "load_examples": usage_examples,
        "primary_split": split_names[0],
        "shard_glob": f"*{output['shard_extension']}",
        "shard_count": f"{sum(shard_counts.values()):,}",
        "shards_dir": output["shards_dir"],
        "split_data_files": split_shards,
        "split_names": ", ".join(f"`{split}`" for split in split_names),
        "split_stats": split_stats,
    }


def _metadata_replacements(summary: DatasetSummary) -> dict[str, str]:
    """Build identity, sample, label, and metadata replacement values."""
    output = DATASET_CONFIG["output"]
    sample_prefix = str(0).zfill(output["sample_prefix_width"])
    return {
        "citation_year": str(datetime.now(UTC).year),
        "dataset_summary": json.dumps(summary, indent=4),
        "sample_metadata": json.dumps(
            {"screen": summary["screen"][0], "safety": summary["safety"][0]},
            indent=4,
        ),
        "image_extensions": ", ".join(f"`{extension}`" for extension in IMAGE_CONFIG["extensions"]),
        "image_keys": ", ".join(f'"{extension.removeprefix(".")}"' for extension in IMAGE_CONFIG["extensions"]) + ",",
        "pretty_name": DATASET_CONFIG["card"]["pretty_name"],
        "repo_id": DATASET_CONFIG["repo"]["id"],
        "safety_values": ", ".join(f"`{value}`" for value in DATASET_CONFIG["source"]["safety_values"]),
        "sample_image_name": f"{sample_prefix}{IMAGE_CONFIG['extensions'][0]}",
        "sample_json_name": f"{sample_prefix}.json",
        "size_category": next(
            (label for limit, label in SIZE_CATEGORIES if summary["total_images"] < limit),
            "n>1M",
        ),
        "total_images": f"{summary['total_images']:,}",
    }


def build_dataset_card(summary: DatasetSummary, options: DatasetBuildOptions) -> str:
    """Build and fully resolve the dataset card template."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {**_split_replacements(summary, options), **_metadata_replacements(summary)}
    card = template
    for name, value in replacements.items():
        card = card.replace(f"{{{{{name}}}}}", value)
    unresolved = sorted(set(PLACEHOLDER_PATTERN.findall(card)))
    if unresolved:
        msg = f"dataset card has unresolved placeholders: {', '.join(unresolved)}"
        raise ValueError(msg)
    return card if card.endswith("\n") else f"{card}\n"
