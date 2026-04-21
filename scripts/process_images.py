from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from configs.data_config import (
    CLASS_IDS,
    CLASS_MAPPING,
    DATASET_NAME,
    DEFAULT_SOURCES,
    RAW_DATASET_DIRS,
    TARGET_SIZE,
    get_dataset_dir,
    get_images_dir,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
KERENBERKE_PREFIX_RE = re.compile(r"([A-Za-z\-]+)")


def crop_center_and_resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = img.size
    new_dim = min(width, height)
    left = (width - new_dim) / 2
    top = (height - new_dim) / 2
    right = (width + new_dim) / 2
    bottom = (height + new_dim) / 2
    return img.crop((left, top, right, bottom)).resize(size, Image.Resampling.LANCZOS)


def normalize_label(raw_label: str) -> str:
    return raw_label.strip().lower()


def resolve_target_class(raw_label: str) -> str:
    return CLASS_MAPPING.get(normalize_label(raw_label), "other")


def iter_standard_source(source_root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for file_path in source_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        items.append((file_path.parent.name, file_path))
    return items


def extract_kerenberke_label(file_path: Path) -> str | None:
    stem = file_path.name.split(".rf.")[0]
    match = KERENBERKE_PREFIX_RE.match(stem)
    if match is None:
        return None
    return match.group(1).lower()


def iter_kerenberke_source(source_root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for file_path in source_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        raw_label = extract_kerenberke_label(file_path)
        if raw_label is None:
            continue
        items.append((raw_label, file_path))
    return items


def collect_source_items(source_name: str, source_root: Path) -> list[tuple[str, Path]]:
    if source_name == "kerenberke":
        return iter_kerenberke_source(source_root)
    return iter_standard_source(source_root)


def build_output_name(source_name: str, source_root: Path, file_path: Path, raw_label: str) -> str:
    relative_parent = "_".join(file_path.relative_to(source_root).parts[:-1])
    safe_parent = relative_parent.replace(" ", "_") if relative_parent else "root"
    safe_label = normalize_label(raw_label).replace(" ", "_")
    return f"{source_name}_{safe_parent}_{safe_label}_{file_path.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a merged 3-class trash dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DATASET_NAME,
        help="Output dataset folder name under data/.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(RAW_DATASET_DIRS.keys()),
        default=list(DEFAULT_SOURCES),
        help="Raw dataset sources to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = get_dataset_dir(args.dataset_name)
    output_images_dir = get_images_dir(args.dataset_name)

    if output_images_dir.exists():
        shutil.rmtree(output_images_dir)

    for class_name in CLASS_IDS:
        (output_images_dir / class_name).mkdir(parents=True, exist_ok=True)

    seen_hashes: set[imagehash.ImageHash] = set()
    saved_by_class: Counter[str] = Counter()
    saved_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    seen_by_source: Counter[str] = Counter()
    duplicate_by_source: Counter[str] = Counter()
    error_count = 0
    missing_sources: list[str] = []

    all_items: list[tuple[str, str, Path, Path]] = []
    for source_name in args.sources:
        source_root = RAW_DATASET_DIRS[source_name]
        if not source_root.exists():
            missing_sources.append(source_name)
            continue
        for raw_label, file_path in collect_source_items(source_name, source_root):
            all_items.append((source_name, raw_label, file_path, source_root))

    for source_name, raw_label, file_path, source_root in tqdm(all_items, desc="Processing images"):
        target_class = resolve_target_class(raw_label)
        seen_by_source[source_name] += 1

        try:
            with Image.open(file_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img_hash = imagehash.phash(img)
                if img_hash in seen_hashes:
                    duplicate_by_source[source_name] += 1
                    continue

                seen_hashes.add(img_hash)
                target_file = output_images_dir / target_class / build_output_name(
                    source_name=source_name,
                    source_root=source_root,
                    file_path=file_path,
                    raw_label=raw_label,
                )
                crop_center_and_resize(img, TARGET_SIZE).save(target_file, "JPEG", quality=90)
                saved_by_class[target_class] += 1
                saved_by_source[source_name][target_class] += 1
        except Exception as exc:
            error_count += 1
            print(f"\nFailed to process {file_path}: {exc}")

    summary = {
        "dataset_name": args.dataset_name,
        "sources": args.sources,
        "missing_sources": missing_sources,
        "target_size": list(TARGET_SIZE),
        "total_seen_images": int(sum(seen_by_source.values())),
        "total_saved_images": int(sum(saved_by_class.values())),
        "duplicate_images_skipped": int(sum(duplicate_by_source.values())),
        "errors": error_count,
        "saved_by_class": dict(sorted(saved_by_class.items())),
        "saved_by_source": {
            source_name: dict(sorted(counter.items()))
            for source_name, counter in sorted(saved_by_source.items())
        },
        "seen_by_source": dict(sorted(seen_by_source.items())),
        "duplicates_by_source": dict(sorted(duplicate_by_source.items())),
    }

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "build_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Built dataset in {dataset_dir}")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
