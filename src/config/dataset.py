"""Dataset build configuration."""

from src.state.dataset import DatasetConfig

DATASET_CONFIG: DatasetConfig = {
    "collection_concurrency": 8,
    "repo": {
        "id": "yapwithai/phone-screenshots",
    },
    "source": {
        "path": "dataset",
        "safety_values": ("safe", "hot", "nsfw", "forbidden"),
    },
    "split": {
        "names": ("train", "val", "test"),
        "default_val_percent": 7.5,
        "default_test_percent": 7.5,
        "max_held_out_percent": 50,
        "min_bucket_size": 3,
    },
    "output": {
        "default_dir": "output/dataset/phone-screenshots",
        "readme_file": "README.md",
        "metadata_file": "dataset_info.json",
        "shards_dir": "data",
        "manifest_jsonl_file": "manifest.jsonl",
        "rejected_jsonl_file": "rejected.jsonl",
        "manifest_parquet_dir": "parquet",
        "manifest_parquet_part_file": "part-00000.parquet",
        "sample_prefix_width": 12,
        "shard_extension": ".tar",
        "shard_index_width": 6,
        "shard_target_bytes": 1073741824,
    },
    "card": {
        "pretty_name": "Screenshot Classifier Dataset",
    },
}
