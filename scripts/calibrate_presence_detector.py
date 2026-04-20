from __future__ import annotations

# Desktop-only calibration utility. This script is for tuning and generating a
# small background reference on a laptop, not for the future ESP runtime.

import argparse
import importlib
from pathlib import Path

import cv2
import numpy as np

from configs.runtime.presence_config import RuntimeConfig
from src.pipeline.presence_detector import PresenceDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_runtime_config(config_name: str) -> RuntimeConfig:
    module_name = config_name
    if module_name.startswith("configs."):
        pass
    elif module_name.startswith("runtime."):
        module_name = f"configs.{module_name}"
    else:
        module_name = f"configs.runtime.{config_name}"

    config_module = importlib.import_module(module_name)
    return config_module.RuntimeConfig()


def collect_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not image_paths:
        raise ValueError(f"No image files found in: {input_dir}")
    return image_paths


def read_frames(image_paths: list[Path], max_frames: int | None = None) -> list[np.ndarray]:
    selected_paths = image_paths[:max_frames] if max_frames is not None else image_paths
    frames: list[np.ndarray] = []

    for image_path in selected_paths:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        frames.append(frame)

    return frames


def save_background(detector: PresenceDetector, output_path: Path) -> None:
    background = detector.background_reference
    if background is None:
        raise RuntimeError("Background reference is not available.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".npy":
        np.save(output_path, background)
        return

    if not cv2.imwrite(str(output_path), background):
        raise ValueError(f"Could not save background preview to: {output_path}")


def save_debug_preview(
    detector: PresenceDetector,
    sample_frame: np.ndarray,
    output_path: Path,
) -> None:
    processed = detector.preprocess(sample_frame)
    background = detector.background_reference
    if background is None:
        raise RuntimeError("Background reference is not available.")

    sample_panel = processed.roi_small
    background_panel = background
    diff_panel = cv2.absdiff(processed.roi_blurred, background)

    stacked = np.concatenate([sample_panel, background_panel, diff_panel], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), stacked):
        raise ValueError(f"Could not save debug preview to: {output_path}")


def print_stats(detector: PresenceDetector, frames: list[np.ndarray]) -> None:
    background = detector.background_reference
    if background is None:
        raise RuntimeError("Background reference is not available.")

    brightness_values: list[float] = []
    diff_values: list[float] = []
    ratio_values: list[float] = []

    for frame in frames:
        processed = detector.preprocess(frame)
        diff = cv2.absdiff(processed.roi_blurred, background)
        mask = detector.build_mask(diff)
        brightness_values.append(float(np.mean(processed.roi_blurred)))
        diff_values.append(float(np.mean(diff)))
        ratio_values.append(float(np.count_nonzero(mask)) / float(mask.size))

    print("=== Presence detector calibration ===")
    print(f"Frames used: {len(frames)}")
    print(f"Background shape: {background.shape}")
    print(
        "Brightness mean/std: "
        f"{np.mean(brightness_values):.2f} / {np.std(brightness_values):.2f}"
    )
    print(
        "Diff mean/std: "
        f"{np.mean(diff_values):.2f} / {np.std(diff_values):.2f}"
    )
    print(
        "Foreground ratio mean/max: "
        f"{np.mean(ratio_values):.5f} / {np.max(ratio_values):.5f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="presence_config",
        help="Runtime config module name from configs/runtime/ (without .py)",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory with empty-scene frames for background calibration",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Where to save the background reference (.npy or image)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional limit for how many frames to use",
    )
    parser.add_argument(
        "--debug_preview_path",
        type=Path,
        default=None,
        help="Optional path for a debug preview PNG",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_runtime_config(args.config)
    image_paths = collect_image_paths(args.input_dir)
    frames = read_frames(image_paths, max_frames=args.max_frames)

    detector = PresenceDetector(config.detector)
    detector.fit_background(frames)

    save_background(detector, args.output_path)
    if args.debug_preview_path is not None:
        save_debug_preview(detector, frames[0], args.debug_preview_path)

    print_stats(detector, frames)
    print(f"Saved background reference to: {args.output_path}")
    if args.debug_preview_path is not None:
        print(f"Saved debug preview to: {args.debug_preview_path}")


if __name__ == "__main__":
    main()
