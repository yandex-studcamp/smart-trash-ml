from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TACO_REPOSITORY_URL = "https://github.com/pedropro/TACO"
USER_AGENT = "smart-trash-ml/esp-anomaly-dataset-builder"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15

DEFAULT_TRAIN_TARGET = 480
DEFAULT_VAL_TARGET = 96
DEFAULT_TEST_NORMAL_TARGET = 96
DEFAULT_TEST_ANOMALY_TARGET = 192
DEFAULT_MAX_CUTOUTS = 160

DEFAULT_TACO_SUPERCATEGORIES = (
    "Aluminium foil",
    "Bottle",
    "Bottle cap",
    "Can",
    "Carton",
    "Cup",
    "Food waste",
    "Glass jar",
    "Lid",
    "Paper",
    "Paper bag",
    "Plastic bag & wrapper",
    "Plastic container",
    "Plastic utensils",
    "Pop tab",
    "Squeezable tube",
    "Straw",
    "Styrofoam piece",
)


@dataclass(frozen=True, slots=True)
class CutoutRecord:
    path: str
    category: str
    supercategory: str
    width: int
    height: int
    annotation_id: int
    image_id: int
    source_url: str


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    split: str
    label_id: int
    file_path: str
    source_image: str
    synthetic_overlay: bool
    overlay_count: int
    overlay_supercategories: str
    overlay_categories: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic anomaly-detection dataset from ESP empty frames. "
            "The script creates train/validation/test CSV splits compatible with "
            "the repository anomaly pipeline and can synthesize anomalies by "
            "overlaying segmented TACO objects."
        ),
    )
    parser.add_argument(
        "--normal-dir",
        type=Path,
        default=Path("data/from_esp/samples"),
        help="Directory with normal empty-scene images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/anomaly_detection_dataset"),
        help="Prepared dataset root directory.",
    )
    parser.add_argument(
        "--taco-cache-dir",
        type=Path,
        default=Path("data/external/taco"),
        help="Directory used to cache TACO annotations, images and cutouts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits and synthesis.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Ratio of source empty frames assigned to train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Ratio of source empty frames assigned to validation.",
    )
    parser.add_argument(
        "--train-target",
        type=int,
        default=DEFAULT_TRAIN_TARGET,
        help="Number of generated train normal images.",
    )
    parser.add_argument(
        "--val-target",
        type=int,
        default=DEFAULT_VAL_TARGET,
        help="Number of generated validation normal images.",
    )
    parser.add_argument(
        "--test-normal-target",
        type=int,
        default=DEFAULT_TEST_NORMAL_TARGET,
        help="Number of generated normal test images.",
    )
    parser.add_argument(
        "--test-anomaly-target",
        type=int,
        default=DEFAULT_TEST_ANOMALY_TARGET,
        help="Number of generated anomaly test images.",
    )
    parser.add_argument(
        "--max-cutouts",
        type=int,
        default=DEFAULT_MAX_CUTOUTS,
        help="Maximum number of TACO object cutouts to extract and cache.",
    )
    parser.add_argument(
        "--cutout-min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum annotation area / image area to keep a cutout.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the existing prepared dataset directory before generation.",
    )
    parser.add_argument(
        "--skip-taco-download",
        action="store_true",
        help="Do not try to download missing TACO assets. Useful when cache already exists.",
    )
    parser.add_argument(
        "--taco-supercategories",
        nargs="+",
        default=list(DEFAULT_TACO_SUPERCATEGORIES),
        help="Allowed TACO supercategories for synthetic object overlays.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.normal_dir = args.normal_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.taco_cache_dir = args.taco_cache_dir.resolve()
    return args


def collect_image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Normal images directory does not exist: {directory}")

    image_paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(image_paths) < 3:
        raise ValueError(
            "At least 3 normal source images are required to create train/validation/test splits.",
        )
    return image_paths


def split_source_images(
    image_paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    shuffled = list(image_paths)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, round(total * train_ratio))
    val_count = max(1, round(total * val_ratio))
    test_count = total - train_count - val_count

    while test_count < 1:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            raise ValueError("Could not allocate at least one source image to each split.")
        test_count = total - train_count - val_count

    train_paths = shuffled[:train_count]
    val_paths = shuffled[train_count:train_count + val_count]
    test_paths = shuffled[train_count + val_count:]

    return {
        "train": sorted(train_paths),
        "validation": sorted(val_paths),
        "test": sorted(test_paths),
    }


def ensure_clean_output(output_dir: Path, clean_output: bool) -> None:
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    (output_dir / "images" / "train" / "normal").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "validation" / "normal").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "test" / "normal").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "test" / "anomaly").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)


def load_taco_annotations(cache_dir: Path, allow_download: bool) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = cache_dir / "annotations.json"

    if not annotations_path.exists():
        if not allow_download:
            raise FileNotFoundError(
                "TACO annotations were not found in cache and downloads are disabled: "
                f"{annotations_path}",
            )
        download_file(TACO_ANNOTATIONS_URL, annotations_path)

    with annotations_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
    except urllib.error.URLError as error:
        raise RuntimeError(f"Failed to download {url}: {error}") from error


def slugify(value: str) -> str:
    sanitized = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    sanitized = "_".join(filter(None, sanitized.split("_")))
    return sanitized or "item"


def maybe_download_taco_image(
    image_meta: dict,
    images_dir: Path,
    allow_download: bool,
) -> Path | None:
    suffix = Path(image_meta.get("flickr_640_url") or image_meta.get("flickr_url") or image_meta["file_name"]).suffix
    image_path = images_dir / f"{image_meta['id']:06d}{suffix.lower() or '.jpg'}"
    if image_path.exists():
        return image_path
    if not allow_download:
        return None

    url = image_meta.get("flickr_640_url") or image_meta.get("flickr_url")
    if not url:
        return None
    try:
        download_file(url, image_path)
    except RuntimeError:
        return None
    return image_path


def load_cached_cutouts(
    manifest_path: Path,
    allowed_supercategories: set[str],
) -> list[CutoutRecord]:
    if not manifest_path.exists():
        return []

    records: list[CutoutRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["supercategory"] not in allowed_supercategories:
                continue
            if not Path(row["path"]).exists():
                continue
            records.append(
                CutoutRecord(
                    path=row["path"],
                    category=row["category"],
                    supercategory=row["supercategory"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                    annotation_id=int(row["annotation_id"]),
                    image_id=int(row["image_id"]),
                    source_url=row["source_url"],
                ),
            )
    return records


def save_cutout_manifest(records: Iterable[CutoutRecord], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "category",
                "supercategory",
                "width",
                "height",
                "annotation_id",
                "image_id",
                "source_url",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def ensure_taco_cutouts(
    cache_dir: Path,
    allowed_supercategories: set[str],
    max_cutouts: int,
    min_area_ratio: float,
    rng: random.Random,
    allow_download: bool,
) -> list[CutoutRecord]:
    manifest_path = cache_dir / "cutouts_manifest.csv"
    cached = load_cached_cutouts(manifest_path, allowed_supercategories)
    if len(cached) >= max_cutouts:
        rng.shuffle(cached)
        return cached[:max_cutouts]

    annotations = load_taco_annotations(cache_dir, allow_download=allow_download)
    categories_by_id = {
        int(category["id"]): {
            "name": str(category["name"]),
            "supercategory": str(category["supercategory"]),
        }
        for category in annotations["categories"]
    }
    images_by_id = {
        int(image["id"]): image
        for image in annotations["images"]
    }

    known_annotation_ids = {record.annotation_id for record in cached}
    candidate_annotations = list(annotations["annotations"])
    rng.shuffle(candidate_annotations)

    images_dir = cache_dir / "images"
    cutouts_dir = cache_dir / "cutouts"
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    collected = list(cached)
    progress = tqdm(candidate_annotations, desc="Preparing TACO cutouts", unit="ann")
    for annotation in progress:
        if len(collected) >= max_cutouts:
            break

        annotation_id = int(annotation["id"])
        if annotation_id in known_annotation_ids:
            continue

        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        category_meta = categories_by_id.get(category_id)
        image_meta = images_by_id.get(image_id)
        if category_meta is None or image_meta is None:
            continue
        if category_meta["supercategory"] not in allowed_supercategories:
            continue

        image_area = float(image_meta["width"]) * float(image_meta["height"])
        if image_area <= 0.0:
            continue

        annotation_area = float(annotation.get("area", 0.0))
        if annotation_area / image_area < min_area_ratio:
            continue

        source_image_path = maybe_download_taco_image(image_meta, images_dir, allow_download=allow_download)
        if source_image_path is None or not source_image_path.exists():
            continue

        record = extract_cutout_from_annotation(
            annotation=annotation,
            image_meta=image_meta,
            category_name=category_meta["name"],
            supercategory=category_meta["supercategory"],
            source_image_path=source_image_path,
            output_dir=cutouts_dir,
        )
        if record is None:
            continue

        collected.append(record)
        known_annotation_ids.add(annotation_id)
        progress.set_postfix(count=len(collected))

    if not collected:
        raise RuntimeError(
            "No usable TACO cutouts were prepared. Check the internet connection or try "
            "loosening the supercategory / area filters.",
        )

    save_cutout_manifest(collected, manifest_path)
    rng.shuffle(collected)
    return collected[:max_cutouts]


def extract_cutout_from_annotation(
    annotation: dict,
    image_meta: dict,
    category_name: str,
    supercategory: str,
    source_image_path: Path,
    output_dir: Path,
) -> CutoutRecord | None:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list) or not segmentation:
        return None

    with Image.open(source_image_path) as source_image_raw:
        source_image = source_image_raw.convert("RGBA")
        source_width, source_height = source_image.size

        original_width = float(image_meta["width"])
        original_height = float(image_meta["height"])
        if original_width <= 0.0 or original_height <= 0.0:
            return None

        scale_x = source_width / original_width
        scale_y = source_height / original_height

        mask = Image.new("L", (source_width, source_height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            scaled_points = [
                (polygon[index] * scale_x, polygon[index + 1] * scale_y)
                for index in range(0, len(polygon), 2)
            ]
            draw.polygon(scaled_points, fill=255)

        bounding_box = mask.getbbox()
        if bounding_box is None:
            return None

        cropped_image = source_image.crop(bounding_box)
        cropped_mask = mask.crop(bounding_box)
        width, height = cropped_image.size
        if min(width, height) < 20:
            return None

        alpha_array = np.asarray(cropped_mask, dtype=np.uint8)
        if alpha_array.size == 0:
            return None

        coverage = float(np.count_nonzero(alpha_array)) / float(alpha_array.size)
        if coverage < 0.10:
            return None

        rgba = cropped_image.copy()
        rgba.putalpha(cropped_mask)

        category_slug = slugify(supercategory)
        output_path = output_dir / category_slug / f"{annotation['image_id']:06d}_{annotation['id']:06d}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(output_path)

        source_url = str(image_meta.get("flickr_640_url") or image_meta.get("flickr_url") or "")
        return CutoutRecord(
            path=str(output_path.resolve()),
            category=category_name,
            supercategory=supercategory,
            width=width,
            height=height,
            annotation_id=int(annotation["id"]),
            image_id=int(annotation["image_id"]),
            source_url=source_url,
        )


def add_gaussian_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    if sigma <= 0.0:
        return image

    array = np.asarray(image, dtype=np.float32)
    noise = np.asarray(
        [rng.gauss(0.0, sigma) for _ in range(array.size)],
        dtype=np.float32,
    ).reshape(array.shape)
    array = np.clip(array + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def apply_jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        return reopened.convert("L")


def augment_background(
    image: Image.Image,
    rng: random.Random,
    profile: str,
) -> Image.Image:
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    fill_value = int(ImageStat.Stat(gray).mean[0])

    if profile == "train":
        max_angle = 5.0
        max_shift = 0.05
        min_scale = 0.95
        max_scale = 1.05
        blur_probability = 0.45
        noise_sigma = 4.0
    else:
        max_angle = 3.0
        max_shift = 0.03
        min_scale = 0.97
        max_scale = 1.03
        blur_probability = 0.30
        noise_sigma = 2.5

    gray = TF.affine(
        gray,
        angle=rng.uniform(-max_angle, max_angle),
        translate=(int(width * rng.uniform(-max_shift, max_shift)), int(height * rng.uniform(-max_shift, max_shift))),
        scale=rng.uniform(min_scale, max_scale),
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=fill_value,
    )

    gray = TF.adjust_brightness(gray, rng.uniform(0.90, 1.10))
    gray = TF.adjust_contrast(gray, rng.uniform(0.90, 1.12))
    gray = TF.adjust_gamma(gray, rng.uniform(0.92, 1.08))

    if rng.random() < blur_probability:
        sigma = rng.uniform(0.15, 0.90 if profile == "train" else 0.60)
        gray = TF.gaussian_blur(gray, kernel_size=3, sigma=sigma)

    if rng.random() < 0.65:
        gray = add_gaussian_noise(gray, sigma=rng.uniform(0.4, noise_sigma), rng=rng)

    if rng.random() < 0.55:
        gray = apply_jpeg_roundtrip(gray, quality=rng.randint(45, 85))

    return gray


def resize_cutout_for_canvas(
    cutout: Image.Image,
    canvas_size: tuple[int, int],
    rng: random.Random,
) -> Image.Image:
    canvas_width, canvas_height = canvas_size
    min_canvas_side = min(canvas_width, canvas_height)

    target_fraction = rng.uniform(0.18, 0.42)
    if rng.random() < 0.20:
        target_fraction = rng.uniform(0.10, 0.20)

    max_object_side = max(1, int(min_canvas_side * target_fraction))
    cutout_width, cutout_height = cutout.size
    scale = max_object_side / float(max(cutout_width, cutout_height))
    resized_width = max(14, int(round(cutout_width * scale)))
    resized_height = max(14, int(round(cutout_height * scale)))

    return cutout.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)


def augment_cutout(
    cutout_path: Path,
    canvas_size: tuple[int, int],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    with Image.open(cutout_path) as cutout_raw:
        cutout = cutout_raw.convert("RGBA")

    if rng.random() < 0.5:
        cutout = ImageOps.mirror(cutout)

    cutout = resize_cutout_for_canvas(cutout, canvas_size, rng)
    rotation = rng.uniform(-30.0, 30.0)
    cutout = cutout.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)

    grayscale = ImageOps.grayscale(cutout.convert("RGB"))
    alpha = cutout.getchannel("A")

    grayscale = TF.adjust_brightness(grayscale, rng.uniform(0.88, 1.12))
    grayscale = TF.adjust_contrast(grayscale, rng.uniform(0.90, 1.15))
    if rng.random() < 0.35:
        grayscale = TF.gaussian_blur(grayscale, kernel_size=3, sigma=rng.uniform(0.10, 0.60))

    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))
    return grayscale, alpha


def match_object_to_background(
    object_gray: Image.Image,
    alpha: Image.Image,
    local_mean: float,
    rng: random.Random,
) -> Image.Image:
    object_array = np.asarray(object_gray, dtype=np.float32)
    alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
    mask = alpha_array > 0.05
    if not np.any(mask):
        return object_gray

    current_mean = float(object_array[mask].mean())
    current_std = float(object_array[mask].std())
    target_mean = np.clip(local_mean * rng.uniform(0.78, 1.18), 20.0, 235.0)
    target_std = max(12.0, current_std * rng.uniform(0.85, 1.20))

    centered = object_array - current_mean
    if current_std > 1e-3:
        centered = centered * (target_std / current_std)
    adjusted = centered + target_mean
    object_array[mask] = np.clip(adjusted[mask], 0.0, 255.0)
    return Image.fromarray(object_array.astype(np.uint8), mode="L")


def choose_object_position(
    canvas_size: tuple[int, int],
    object_size: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    object_width, object_height = object_size

    max_x = max(0, canvas_width - object_width)
    max_y = max(0, canvas_height - object_height)

    left_bias = 0.10 * max_x
    right_bias = 0.90 * max_x
    top_bias = 0.18 * max_y
    bottom_bias = 0.92 * max_y

    x = int(round(rng.uniform(left_bias, right_bias if right_bias > left_bias else max_x)))
    y = int(round(rng.uniform(top_bias, bottom_bias if bottom_bias > top_bias else max_y)))
    return min(max_x, x), min(max_y, y)


def paste_shadow(
    canvas: Image.Image,
    alpha: Image.Image,
    position: tuple[int, int],
    local_mean: float,
    rng: random.Random,
) -> None:
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.4, 2.8)))
    shadow_alpha = shadow_alpha.point(lambda value: int(value * rng.uniform(0.18, 0.32)))
    offset = (rng.randint(1, 3), rng.randint(1, 3))
    shadow_position = (
        min(canvas.width - 1, position[0] + offset[0]),
        min(canvas.height - 1, position[1] + offset[1]),
    )
    shadow_value = max(0, int(local_mean - rng.uniform(18.0, 42.0)))
    canvas.paste(shadow_value, shadow_position, shadow_alpha)


def overlay_cutout(
    canvas: Image.Image,
    cutout_record: CutoutRecord,
    rng: random.Random,
) -> dict[str, str]:
    object_gray, alpha = augment_cutout(Path(cutout_record.path), canvas.size, rng)
    if object_gray.width >= canvas.width or object_gray.height >= canvas.height:
        resize_factor = min(
            (canvas.width - 8) / float(max(object_gray.width, 1)),
            (canvas.height - 8) / float(max(object_gray.height, 1)),
        )
        resize_factor = max(0.10, resize_factor)
        new_size = (
            max(12, int(round(object_gray.width * resize_factor))),
            max(12, int(round(object_gray.height * resize_factor))),
        )
        object_gray = object_gray.resize(new_size, resample=Image.Resampling.BICUBIC)
        alpha = alpha.resize(new_size, resample=Image.Resampling.BICUBIC)

    position = choose_object_position(canvas.size, object_gray.size, rng)
    local_patch = canvas.crop((position[0], position[1], position[0] + object_gray.width, position[1] + object_gray.height))
    local_mean = float(ImageStat.Stat(local_patch).mean[0]) if local_patch.size[0] > 0 and local_patch.size[1] > 0 else 128.0

    object_gray = match_object_to_background(object_gray, alpha, local_mean=local_mean, rng=rng)
    opacity = rng.uniform(0.80, 0.96)
    alpha = alpha.point(lambda value: int(value * opacity))

    paste_shadow(canvas, alpha, position, local_mean=local_mean, rng=rng)
    canvas.paste(object_gray, position, alpha)

    return {
        "supercategory": cutout_record.supercategory,
        "category": cutout_record.category,
    }


def build_normal_variant(
    source_path: Path,
    rng: random.Random,
    profile: str,
) -> Image.Image:
    with Image.open(source_path) as source_image:
        return augment_background(source_image, rng=rng, profile=profile)


def build_anomaly_variant(
    source_path: Path,
    cutouts: list[CutoutRecord],
    rng: random.Random,
) -> tuple[Image.Image, list[dict[str, str]]]:
    canvas = build_normal_variant(source_path, rng=rng, profile="eval")
    overlay_count = 1 if rng.random() < 0.82 else 2
    overlays: list[dict[str, str]] = []

    for _ in range(overlay_count):
        cutout_record = rng.choice(cutouts)
        overlay_meta = overlay_cutout(canvas, cutout_record=cutout_record, rng=rng)
        overlays.append(overlay_meta)

    if rng.random() < 0.45:
        canvas = TF.gaussian_blur(canvas, kernel_size=3, sigma=rng.uniform(0.10, 0.45))
    if rng.random() < 0.70:
        canvas = apply_jpeg_roundtrip(canvas, quality=rng.randint(42, 78))
    if rng.random() < 0.55:
        canvas = add_gaussian_noise(canvas, sigma=rng.uniform(0.25, 1.8), rng=rng)

    return canvas, overlays


def save_generated_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="JPEG", quality=90, optimize=True)


def generate_split_samples(
    split_name: str,
    label_name: str,
    source_paths: list[Path],
    target_count: int,
    output_dir: Path,
    rng: random.Random,
    cutouts: list[CutoutRecord] | None = None,
) -> list[GeneratedSample]:
    if target_count <= 0:
        return []
    if not source_paths:
        raise ValueError(f"No source images available for split: {split_name}")

    generated: list[GeneratedSample] = []
    description = f"Generating {split_name}/{label_name}"
    for index in tqdm(range(target_count), desc=description, unit="img"):
        source_path = rng.choice(source_paths)
        source_name = source_path.name

        if label_name == "normal":
            profile = "train" if split_name == "train" else "eval"
            image = build_normal_variant(source_path, rng=rng, profile=profile)
            overlays: list[dict[str, str]] = []
        else:
            if cutouts is None:
                raise ValueError("Synthetic anomaly generation requires cutouts.")
            image, overlays = build_anomaly_variant(source_path, cutouts=cutouts, rng=rng)

        relative_path = Path("images") / split_name / label_name / f"{split_name}_{label_name}_{index:05d}.jpg"
        absolute_path = output_dir / relative_path
        save_generated_image(image, absolute_path)

        generated.append(
            GeneratedSample(
                split=split_name,
                label_id=0 if label_name == "normal" else 1,
                file_path=str(relative_path.as_posix()),
                source_image=source_name,
                synthetic_overlay=label_name == "anomaly",
                overlay_count=len(overlays),
                overlay_supercategories="|".join(item["supercategory"] for item in overlays),
                overlay_categories="|".join(item["category"] for item in overlays),
            ),
        )

    return generated


def write_split_csv(samples: list[GeneratedSample], output_path: Path) -> None:
    rows = [
        {
            "file_path": sample.file_path,
            "label_id": sample.label_id,
            "split": sample.split,
            "source_image": sample.source_image,
            "synthetic_overlay": sample.synthetic_overlay,
            "overlay_count": sample.overlay_count,
            "overlay_supercategories": sample.overlay_supercategories,
            "overlay_categories": sample.overlay_categories,
        }
        for sample in samples
    ]
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)


def save_metadata(
    output_dir: Path,
    normal_dir: Path,
    split_sources: dict[str, list[Path]],
    generated_counts: dict[str, int],
    cutouts: list[CutoutRecord],
    args: argparse.Namespace,
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "normal_source_dir": str(normal_dir),
        "dataset_root": str(output_dir),
        "taco_repository_url": TACO_REPOSITORY_URL,
        "taco_annotations_url": TACO_ANNOTATIONS_URL,
        "allowed_taco_supercategories": sorted(args.taco_supercategories),
        "source_split_counts": {split: len(paths) for split, paths in split_sources.items()},
        "source_split_files": {
            split: [path.name for path in paths]
            for split, paths in split_sources.items()
        },
        "generated_counts": generated_counts,
        "taco_cutout_count": len(cutouts),
        "builder_args": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "train_target": args.train_target,
            "val_target": args.val_target,
            "test_normal_target": args.test_normal_target,
            "test_anomaly_target": args.test_anomaly_target,
            "max_cutouts": args.max_cutouts,
            "cutout_min_area_ratio": args.cutout_min_area_ratio,
        },
    }

    metadata_path = output_dir / "metadata" / "dataset_manifest.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def main() -> None:
    args = resolve_paths(parse_args())
    rng = random.Random(args.seed)

    normal_images = collect_image_paths(args.normal_dir)
    split_sources = split_source_images(
        image_paths=normal_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        rng=rng,
    )

    ensure_clean_output(args.output_dir, clean_output=args.clean_output)
    cutouts = ensure_taco_cutouts(
        cache_dir=args.taco_cache_dir,
        allowed_supercategories=set(args.taco_supercategories),
        max_cutouts=args.max_cutouts,
        min_area_ratio=args.cutout_min_area_ratio,
        rng=rng,
        allow_download=not args.skip_taco_download,
    )

    train_samples = generate_split_samples(
        split_name="train",
        label_name="normal",
        source_paths=split_sources["train"],
        target_count=args.train_target,
        output_dir=args.output_dir,
        rng=rng,
    )
    validation_samples = generate_split_samples(
        split_name="validation",
        label_name="normal",
        source_paths=split_sources["validation"],
        target_count=args.val_target,
        output_dir=args.output_dir,
        rng=rng,
    )
    test_normal_samples = generate_split_samples(
        split_name="test",
        label_name="normal",
        source_paths=split_sources["test"],
        target_count=args.test_normal_target,
        output_dir=args.output_dir,
        rng=rng,
    )
    test_anomaly_samples = generate_split_samples(
        split_name="test",
        label_name="anomaly",
        source_paths=split_sources["test"],
        target_count=args.test_anomaly_target,
        output_dir=args.output_dir,
        rng=rng,
        cutouts=cutouts,
    )

    write_split_csv(train_samples, args.output_dir / "labels" / "train.csv")
    write_split_csv(validation_samples, args.output_dir / "labels" / "validation.csv")
    write_split_csv(test_normal_samples + test_anomaly_samples, args.output_dir / "labels" / "test.csv")

    generated_counts = {
        "train_normal": len(train_samples),
        "validation_normal": len(validation_samples),
        "test_normal": len(test_normal_samples),
        "test_anomaly": len(test_anomaly_samples),
    }
    save_metadata(
        output_dir=args.output_dir,
        normal_dir=args.normal_dir,
        split_sources=split_sources,
        generated_counts=generated_counts,
        cutouts=cutouts,
        args=args,
    )

    print("=== ESP anomaly dataset prepared ===")
    print(f"Normal source images: {len(normal_images)}")
    print(
        "Base split sizes: "
        f"train={len(split_sources['train'])}, "
        f"validation={len(split_sources['validation'])}, "
        f"test={len(split_sources['test'])}",
    )
    print(
        "Generated samples: "
        f"train_normal={generated_counts['train_normal']}, "
        f"validation_normal={generated_counts['validation_normal']}, "
        f"test_normal={generated_counts['test_normal']}, "
        f"test_anomaly={generated_counts['test_anomaly']}",
    )
    print(f"TACO cutouts available: {len(cutouts)}")
    print(f"Dataset root: {args.output_dir}")
    print(f"Train CSV: {args.output_dir / 'labels' / 'train.csv'}")
    print(f"Validation CSV: {args.output_dir / 'labels' / 'validation.csv'}")
    print(f"Test CSV: {args.output_dir / 'labels' / 'test.csv'}")


if __name__ == "__main__":
    main()
