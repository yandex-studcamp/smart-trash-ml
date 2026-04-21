from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.data_config import (
    CLASS_IDS,
    DATASET_NAME,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
    get_dataset_dir,
    get_images_dir,
    get_labels_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/valid/test CSV splits.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DATASET_NAME,
        help="Dataset folder name under data/.",
    )
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO)
    parser.add_argument("--test_ratio", type=float, default=TEST_RATIO)
    return parser.parse_args()


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")


def collect_records(images_dir) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for class_name in sorted(CLASS_IDS.keys()):
        class_dir = images_dir / class_name
        if not class_dir.exists():
            continue
        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue
            records.append(
                {
                    "file_path": f"images/{class_name}/{file_path.name}",
                    "label_name": class_name,
                    "label_id": CLASS_IDS[class_name],
                }
            )
    return records


def summarize_split(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        "num_samples": int(len(dataframe)),
        "class_counts": {
            class_name: int(count)
            for class_name, count in dataframe["label_name"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    dataset_dir = get_dataset_dir(args.dataset_name)
    images_dir = get_images_dir(args.dataset_name)
    labels_dir = get_labels_dir(args.dataset_name)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    records = collect_records(images_dir)
    if not records:
        raise ValueError(f"No images found in {images_dir}")

    dataframe = pd.DataFrame(records)
    class_counts = dataframe["label_id"].value_counts()
    if (class_counts < 2).any():
        raise ValueError("Each class needs at least 2 samples for stratified splitting.")

    labels_dir.mkdir(parents=True, exist_ok=True)

    holdout_ratio = args.valid_ratio + args.test_ratio
    train_df, holdout_df = train_test_split(
        dataframe,
        test_size=holdout_ratio,
        random_state=42,
        stratify=dataframe["label_id"],
    )
    valid_df, test_df = train_test_split(
        holdout_df,
        test_size=(args.test_ratio / holdout_ratio),
        random_state=42,
        stratify=holdout_df["label_id"],
    )

    train_df.to_csv(labels_dir / "train.csv", index=False)
    valid_df.to_csv(labels_dir / "valid.csv", index=False)
    test_df.to_csv(labels_dir / "test.csv", index=False)

    summary = {
        "dataset_name": args.dataset_name,
        "ratios": {
            "train": args.train_ratio,
            "valid": args.valid_ratio,
            "test": args.test_ratio,
        },
        "total_samples": int(len(dataframe)),
        "train": summarize_split(train_df),
        "valid": summarize_split(valid_df),
        "test": summarize_split(test_df),
    }

    with (dataset_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Created splits in {labels_dir}")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
