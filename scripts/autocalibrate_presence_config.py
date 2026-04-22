from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from configs.runtime.presence_config import (
    PresenceDetectorConfig,
    ROIConfig,
    RuntimeConfig,
    RuntimePipelineConfig,
)
from src.pipeline.presence_detector import PresenceDetector

# Desktop-only autocalibration utility. The goal is to bootstrap a robust
# presence-detector config from normal/empty frames. The resulting config is a
# starting point; final validation against real object/anomaly frames is still
# recommended before shipping to ESP.

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime" / "presence_config_autocalibrated.py"
DEFAULT_BACKGROUND_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "presence" / "background_reference_autocalibrated.npy"
DEFAULT_BACKGROUND_PREVIEW_PATH = PROJECT_ROOT / "experiments" / "presence" / "background_preview_autocalibrated.png"
DEFAULT_ROI_PREVIEW_PATH = PROJECT_ROOT / "experiments" / "presence" / "roi_preview_autocalibrated.png"
DEFAULT_THRESHOLD_PLOT_PATH = PROJECT_ROOT / "experiments" / "presence" / "threshold_sweep_autocalibrated.png"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "experiments" / "presence" / "autocalibration_report.json"


@dataclass(slots=True)
class ThresholdSweepRow:
    threshold: int
    ratio_mean: float
    ratio_quantile: float
    ratio_max: float


@dataclass(slots=True)
class CalibrationArtifacts:
    selected_roi: ROIConfig
    detector_config: PresenceDetectorConfig
    background_reference: np.ndarray
    normal_brightness_values: np.ndarray
    normal_ratios_at_selected_threshold: np.ndarray
    threshold_sweep: list[ThresholdSweepRow]
    roi_search_summary: dict[str, Any]
    normal_static_summary: dict[str, Any]
    anomaly_static_summary: dict[str, Any] | None


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
    frame_shape: tuple[int, ...] | None = None

    for image_path in selected_paths:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        if frame_shape is None:
            frame_shape = frame.shape
        elif frame.shape != frame_shape:
            raise ValueError(
                "All calibration frames must have the same shape. "
                f"Expected {frame_shape}, got {frame.shape} for {image_path}.",
            )

        frames.append(frame)

    if not frames:
        raise ValueError("No frames were loaded for calibration.")
    return frames


def clamp_roi_to_frame(roi: ROIConfig, frame_width: int, frame_height: int) -> ROIConfig:
    width = min(roi.w, frame_width)
    height = min(roi.h, frame_height)
    x = min(max(0, roi.x), max(0, frame_width - width))
    y = min(max(0, roi.y), max(0, frame_height - height))
    return ROIConfig(x=x, y=y, w=width, h=height)


def clone_detector_config(
        base_config: PresenceDetectorConfig,
        *,
        roi: ROIConfig | None = None,
        diff_threshold: int | None = None,
        min_foreground_ratio: float | None = None,
        min_brightness: float | None = None,
        max_brightness: float | None = None,
) -> PresenceDetectorConfig:
    return PresenceDetectorConfig(
        roi=roi or base_config.roi,
        processing_size=base_config.processing_size,
        blur_kernel=base_config.blur_kernel,
        diff_threshold=base_config.diff_threshold if diff_threshold is None else diff_threshold,
        min_foreground_ratio=(
            base_config.min_foreground_ratio
            if min_foreground_ratio is None
            else min_foreground_ratio
        ),
        enter_frames=base_config.enter_frames,
        exit_frames=base_config.exit_frames,
        morph_kernel_size=base_config.morph_kernel_size,
        morph_open_iterations=base_config.morph_open_iterations,
        morph_close_iterations=base_config.morph_close_iterations,
        background_alpha=base_config.background_alpha,
        use_adaptive_background=base_config.use_adaptive_background,
        input_color_order=base_config.input_color_order,
        min_brightness=base_config.min_brightness if min_brightness is None else min_brightness,
        max_brightness=base_config.max_brightness if max_brightness is None else max_brightness,
    )


def prepare_blurred_stack(
        frames: list[np.ndarray],
        detector_config: PresenceDetectorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector = PresenceDetector(detector_config)
    roi_color_stack: list[np.ndarray] = []
    roi_small_stack: list[np.ndarray] = []
    roi_blurred_stack: list[np.ndarray] = []

    for frame in frames:
        prepared = detector.preprocess(frame)
        roi_color_stack.append(prepared.roi_color)
        roi_small_stack.append(prepared.roi_small)
        roi_blurred_stack.append(prepared.roi_blurred)

    return (
        np.stack(roi_color_stack, axis=0),
        np.stack(roi_small_stack, axis=0),
        np.stack(roi_blurred_stack, axis=0),
    )


def compute_background_reference(roi_blurred_stack: np.ndarray) -> np.ndarray:
    background = np.mean(roi_blurred_stack.astype(np.float32), axis=0)
    return np.clip(np.rint(background), 0, 255).astype(np.uint8)


def build_mask_with_config(
        diff: np.ndarray,
        detector_config: PresenceDetectorConfig,
        threshold: int,
) -> np.ndarray:
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (detector_config.morph_kernel_size, detector_config.morph_kernel_size),
    )

    if detector_config.morph_open_iterations > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            morph_kernel,
            iterations=detector_config.morph_open_iterations,
        )
    if detector_config.morph_close_iterations > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            morph_kernel,
            iterations=detector_config.morph_close_iterations,
        )

    return mask


def score_roi_candidate(roi_blurred_stack: np.ndarray) -> dict[str, float]:
    background = compute_background_reference(roi_blurred_stack)
    diffs = [
        cv2.absdiff(frame, background).astype(np.float32)
        for frame in roi_blurred_stack
    ]
    diff_stack = np.stack(diffs, axis=0)
    return {
        "score": float(np.quantile(diff_stack, 0.995)),
        "diff_mean": float(np.mean(diff_stack)),
        "diff_quantile_995": float(np.quantile(diff_stack, 0.995)),
        "diff_max": float(np.max(diff_stack)),
    }


def iter_roi_candidates(
        base_roi: ROIConfig,
        frame_width: int,
        frame_height: int,
        *,
        radius_x: int,
        radius_y: int,
        step: int,
) -> Iterable[ROIConfig]:
    min_x = max(0, base_roi.x - radius_x)
    max_x = min(frame_width - base_roi.w, base_roi.x + radius_x)
    min_y = max(0, base_roi.y - radius_y)
    max_y = min(frame_height - base_roi.h, base_roi.y + radius_y)

    yielded: set[tuple[int, int, int, int]] = set()
    for x in range(min_x, max_x + 1, step):
        for y in range(min_y, max_y + 1, step):
            roi = ROIConfig(x=x, y=y, w=base_roi.w, h=base_roi.h)
            key = roi.as_xywh()
            if key in yielded:
                continue
            yielded.add(key)
            yield roi

    base_key = base_roi.as_xywh()
    if base_key not in yielded:
        yield base_roi


def select_roi(
        frames: list[np.ndarray],
        base_detector_config: PresenceDetectorConfig,
        *,
        refine_roi: bool,
        radius_x: int,
        radius_y: int,
        step: int,
) -> tuple[ROIConfig, dict[str, Any]]:
    frame_height, frame_width = frames[0].shape[:2]
    base_roi = clamp_roi_to_frame(base_detector_config.roi, frame_width, frame_height)

    if not refine_roi:
        return base_roi, {
            "mode": "base_only",
            "base_roi": list(base_roi.as_xywh()),
            "selected_roi": list(base_roi.as_xywh()),
            "candidates_evaluated": 1,
        }

    best_roi = base_roi
    best_summary: dict[str, float] | None = None
    candidates_evaluated = 0

    for roi in iter_roi_candidates(
            base_roi,
            frame_width,
            frame_height,
            radius_x=radius_x,
            radius_y=radius_y,
            step=step,
    ):
        detector_config = clone_detector_config(base_detector_config, roi=roi)
        _, _, roi_blurred_stack = prepare_blurred_stack(frames, detector_config)
        summary = score_roi_candidate(roi_blurred_stack)
        candidates_evaluated += 1

        if best_summary is None or summary["score"] < best_summary["score"]:
            best_roi = roi
            best_summary = summary

    if best_summary is None:
        raise RuntimeError("ROI selection failed to evaluate any candidates.")

    return best_roi, {
        "mode": "local_search",
        "base_roi": list(base_roi.as_xywh()),
        "selected_roi": list(best_roi.as_xywh()),
        "candidates_evaluated": candidates_evaluated,
        "selected_score": best_summary["score"],
        "selected_diff_mean": best_summary["diff_mean"],
        "selected_diff_quantile_995": best_summary["diff_quantile_995"],
        "selected_diff_max": best_summary["diff_max"],
    }


def summarize_brightness(
        brightness_values: np.ndarray,
        *,
        quantile_low: float,
        quantile_high: float,
        margin: float,
) -> tuple[float, float, dict[str, float]]:
    low = max(0.0, float(np.quantile(brightness_values, quantile_low)) - margin)
    high = min(255.0, float(np.quantile(brightness_values, quantile_high)) + margin)

    low = math.floor(low)
    high = math.ceil(high)

    return low, high, {
        "mean": float(np.mean(brightness_values)),
        "std": float(np.std(brightness_values)),
        "min": float(np.min(brightness_values)),
        "max": float(np.max(brightness_values)),
        "quantile_low": float(np.quantile(brightness_values, quantile_low)),
        "quantile_high": float(np.quantile(brightness_values, quantile_high)),
        "selected_min_brightness": float(low),
        "selected_max_brightness": float(high),
    }


def sweep_diff_thresholds(
        roi_blurred_stack: np.ndarray,
        background_reference: np.ndarray,
        detector_config: PresenceDetectorConfig,
        *,
        threshold_min: int,
        threshold_max: int,
        threshold_step: int,
        ratio_quantile: float,
) -> list[ThresholdSweepRow]:
    rows: list[ThresholdSweepRow] = []
    for threshold in range(threshold_min, threshold_max + 1, threshold_step):
        ratios: list[float] = []
        for roi_blurred in roi_blurred_stack:
            diff = cv2.absdiff(roi_blurred, background_reference)
            mask = build_mask_with_config(diff, detector_config, threshold)
            ratios.append(float(np.count_nonzero(mask)) / float(mask.size))

        ratio_array = np.asarray(ratios, dtype=np.float32)
        rows.append(
            ThresholdSweepRow(
                threshold=threshold,
                ratio_mean=float(np.mean(ratio_array)),
                ratio_quantile=float(np.quantile(ratio_array, ratio_quantile)),
                ratio_max=float(np.max(ratio_array)),
            ),
        )

    return rows


def select_thresholds(
        threshold_sweep: list[ThresholdSweepRow],
        *,
        target_empty_ratio: float,
        ratio_margin: float,
) -> tuple[int, float, ThresholdSweepRow]:
    acceptable_rows = [
        row
        for row in threshold_sweep
        if row.ratio_quantile <= target_empty_ratio
    ]
    if acceptable_rows:
        selected_row = acceptable_rows[0]
    else:
        selected_row = min(
            threshold_sweep,
            key=lambda row: (row.ratio_quantile, row.ratio_mean, row.threshold),
        )

    min_foreground_ratio = min(
        1.0,
        max(
            selected_row.ratio_quantile + ratio_margin,
            selected_row.ratio_mean + ratio_margin,
            target_empty_ratio + ratio_margin,
            0.003,
        ),
    )
    return selected_row.threshold, min_foreground_ratio, selected_row


def evaluate_static_signal(
        frames: list[np.ndarray],
        detector_config: PresenceDetectorConfig,
        background_reference: np.ndarray,
) -> dict[str, Any]:
    detector = PresenceDetector(detector_config)
    detector.set_background_reference(background_reference)

    brightness_values: list[float] = []
    foreground_ratios: list[float] = []
    signal_flags: list[bool] = []

    for frame in frames:
        prepared = detector.preprocess(frame)
        brightness = float(np.mean(prepared.roi_blurred))
        diff = cv2.absdiff(prepared.roi_blurred, background_reference)
        mask = build_mask_with_config(diff, detector_config, detector_config.diff_threshold)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        signal_active = (
                detector_config.min_brightness <= brightness <= detector_config.max_brightness
                and ratio >= detector_config.min_foreground_ratio
        )

        brightness_values.append(brightness)
        foreground_ratios.append(ratio)
        signal_flags.append(signal_active)

    brightness_array = np.asarray(brightness_values, dtype=np.float32)
    ratio_array = np.asarray(foreground_ratios, dtype=np.float32)
    signal_array = np.asarray(signal_flags, dtype=np.bool_)

    return {
        "frames": int(len(frames)),
        "signal_rate": float(np.mean(signal_array.astype(np.float32))),
        "brightness_mean": float(np.mean(brightness_array)),
        "brightness_quantile_01": float(np.quantile(brightness_array, 0.01)),
        "brightness_quantile_99": float(np.quantile(brightness_array, 0.99)),
        "foreground_ratio_mean": float(np.mean(ratio_array)),
        "foreground_ratio_quantile_95": float(np.quantile(ratio_array, 0.95)),
        "foreground_ratio_quantile_99": float(np.quantile(ratio_array, 0.99)),
        "foreground_ratio_max": float(np.max(ratio_array)),
    }


def save_background(background_reference: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, background_reference)


def save_background_preview(
        roi_small_stack: np.ndarray,
        background_reference: np.ndarray,
        output_path: Path,
) -> None:
    sample_panel = roi_small_stack[0]
    diff_panel = cv2.absdiff(sample_panel, background_reference)
    stacked = np.concatenate([sample_panel, background_reference, diff_panel], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), stacked):
        raise ValueError(f"Could not save background preview to: {output_path}")


def save_roi_preview(
        sample_frame: np.ndarray,
        base_roi: ROIConfig,
        selected_roi: ROIConfig,
        output_path: Path,
) -> None:
    preview = np.array(sample_frame, copy=True)

    x, y, width, height = base_roi.as_xywh()
    cv2.rectangle(preview, (x, y), (x + width, y + height), (255, 255, 0), 2)
    cv2.putText(preview, "base_roi", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    x, y, width, height = selected_roi.as_xywh()
    cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(preview, "selected_roi", (x, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise ValueError(f"Could not save ROI preview to: {output_path}")


def save_threshold_plot(
        threshold_sweep: list[ThresholdSweepRow],
        selected_threshold: int,
        selected_min_foreground_ratio: float,
        output_path: Path,
) -> None:
    thresholds = [row.threshold for row in threshold_sweep]
    ratio_quantiles = [row.ratio_quantile for row in threshold_sweep]
    ratio_means = [row.ratio_mean for row in threshold_sweep]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, ratio_quantiles, marker="o", label="empty ratio quantile")
    ax.plot(thresholds, ratio_means, marker="o", label="empty ratio mean")
    ax.axvline(selected_threshold, color="green", linestyle="--", label=f"selected diff_threshold={selected_threshold}")
    ax.axhline(
        selected_min_foreground_ratio,
        color="crimson",
        linestyle="--",
        label=f"selected min_foreground_ratio={selected_min_foreground_ratio:.4f}",
    )
    ax.set_xlabel("diff_threshold")
    ax.set_ylabel("foreground_ratio on normal frames")
    ax.set_title("Presence detector threshold sweep")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def serialize_threshold_sweep(threshold_sweep: list[ThresholdSweepRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in threshold_sweep]


def render_config_module(
        detector_config: PresenceDetectorConfig,
        pipeline_config: RuntimePipelineConfig,
) -> str:
    roi = detector_config.roi
    return (
        "from dataclasses import dataclass, field\n"
        "\n"
        "from configs.runtime.presence_config import ROIConfig, PresenceDetectorConfig, RuntimePipelineConfig\n"
        "\n"
        "\n"
        "@dataclass(frozen=True, slots=True)\n"
        "class RuntimeConfig:\n"
        "    detector: PresenceDetectorConfig = field(\n"
        "        default_factory=lambda: PresenceDetectorConfig(\n"
        f"            roi=ROIConfig(x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}),\n"
        f"            processing_size={detector_config.processing_size},\n"
        f"            blur_kernel={detector_config.blur_kernel},\n"
        f"            diff_threshold={detector_config.diff_threshold},\n"
        f"            min_foreground_ratio={detector_config.min_foreground_ratio:.6f},\n"
        f"            enter_frames={detector_config.enter_frames},\n"
        f"            exit_frames={detector_config.exit_frames},\n"
        f"            morph_kernel_size={detector_config.morph_kernel_size},\n"
        f"            morph_open_iterations={detector_config.morph_open_iterations},\n"
        f"            morph_close_iterations={detector_config.morph_close_iterations},\n"
        f"            background_alpha={detector_config.background_alpha:.6f},\n"
        f"            use_adaptive_background={detector_config.use_adaptive_background},\n"
        f"            input_color_order={detector_config.input_color_order!r},\n"
        f"            min_brightness={detector_config.min_brightness:.1f},\n"
        f"            max_brightness={detector_config.max_brightness:.1f},\n"
        "        )\n"
        "    )\n"
        "    pipeline: RuntimePipelineConfig = field(\n"
        "        default_factory=lambda: RuntimePipelineConfig(\n"
        f"            classifier_roi={_render_optional_roi(pipeline_config.classifier_roi)},\n"
        f"            classifier_input_size={pipeline_config.classifier_input_size},\n"
        f"            collect_frames_for_selection={pipeline_config.collect_frames_for_selection},\n"
        f"            cooldown_frames={pipeline_config.cooldown_frames},\n"
        "        )\n"
        "    )\n"
    )


def _render_optional_roi(roi: ROIConfig | None) -> str:
    if roi is None:
        return "None"
    return f"ROIConfig(x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h})"


def save_config_module(
        detector_config: PresenceDetectorConfig,
        pipeline_config: RuntimePipelineConfig,
        output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_config_module(detector_config, pipeline_config),
        encoding="utf-8",
    )


def save_report(
        report_payload: dict[str, Any],
        output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def autocalibrate_from_normals(
        normal_frames: list[np.ndarray],
        base_runtime_config: RuntimeConfig,
        *,
        refine_roi: bool,
        roi_search_radius_x: int,
        roi_search_radius_y: int,
        roi_search_step: int,
        threshold_min: int,
        threshold_max: int,
        threshold_step: int,
        empty_ratio_quantile: float,
        target_empty_ratio: float,
        ratio_margin: float,
        brightness_quantile_low: float,
        brightness_quantile_high: float,
        brightness_margin: float,
        anomaly_frames: list[np.ndarray] | None,
) -> CalibrationArtifacts:
    selected_roi, roi_search_summary = select_roi(
        normal_frames,
        base_runtime_config.detector,
        refine_roi=refine_roi,
        radius_x=roi_search_radius_x,
        radius_y=roi_search_radius_y,
        step=roi_search_step,
    )

    roi_detector_config = clone_detector_config(
        base_runtime_config.detector,
        roi=selected_roi,
    )
    _, roi_small_stack, roi_blurred_stack = prepare_blurred_stack(normal_frames, roi_detector_config)
    background_reference = compute_background_reference(roi_blurred_stack)

    brightness_values = np.mean(roi_blurred_stack.astype(np.float32), axis=(1, 2))
    min_brightness, max_brightness, brightness_summary = summarize_brightness(
        brightness_values,
        quantile_low=brightness_quantile_low,
        quantile_high=brightness_quantile_high,
        margin=brightness_margin,
    )

    brightness_detector_config = clone_detector_config(
        roi_detector_config,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
    )
    threshold_sweep = sweep_diff_thresholds(
        roi_blurred_stack,
        background_reference,
        brightness_detector_config,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        ratio_quantile=empty_ratio_quantile,
    )
    selected_diff_threshold, selected_min_foreground_ratio, selected_threshold_row = select_thresholds(
        threshold_sweep,
        target_empty_ratio=target_empty_ratio,
        ratio_margin=ratio_margin,
    )

    final_detector_config = clone_detector_config(
        brightness_detector_config,
        diff_threshold=selected_diff_threshold,
        min_foreground_ratio=selected_min_foreground_ratio,
    )
    normal_static_summary = evaluate_static_signal(
        normal_frames,
        final_detector_config,
        background_reference,
    )
    normal_static_summary["brightness_summary"] = brightness_summary
    normal_static_summary["selected_threshold_row"] = asdict(selected_threshold_row)

    anomaly_static_summary = None
    if anomaly_frames:
        anomaly_static_summary = evaluate_static_signal(
            anomaly_frames,
            final_detector_config,
            background_reference,
        )

    selected_threshold_ratios = np.asarray(
        [
            float(np.count_nonzero(build_mask_with_config(
                cv2.absdiff(frame, background_reference),
                final_detector_config,
                final_detector_config.diff_threshold,
            ))) / float(frame.size)
            for frame in roi_blurred_stack
        ],
        dtype=np.float32,
    )

    return CalibrationArtifacts(
        selected_roi=selected_roi,
        detector_config=final_detector_config,
        background_reference=background_reference,
        normal_brightness_values=np.asarray(brightness_values, dtype=np.float32),
        normal_ratios_at_selected_threshold=selected_threshold_ratios,
        threshold_sweep=threshold_sweep,
        roi_search_summary=roi_search_summary,
        normal_static_summary=normal_static_summary,
        anomaly_static_summary=anomaly_static_summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Autocalibrate the presence-detector config from a directory of normal/empty frames. "
            "This script estimates a stable ROI refinement, brightness bounds, diff_threshold, "
            "and min_foreground_ratio, then saves a generated runtime config module."
        ),
    )
    parser.add_argument(
        "normal_dir",
        type=Path,
        help="Directory with normal/empty frames from the real camera setup.",
    )
    parser.add_argument(
        "--anomaly_dir",
        type=Path,
        default=None,
        help="Optional directory with object/anomaly frames for post-fit validation only.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="presence_config",
        help="Base runtime config module name from configs/runtime/ (without .py).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional limit for how many normal frames to use.",
    )
    parser.add_argument(
        "--output_config_path",
        type=Path,
        default=DEFAULT_OUTPUT_CONFIG_PATH,
        help="Where to save the generated runtime config .py module.",
    )
    parser.add_argument(
        "--background_output_path",
        type=Path,
        default=DEFAULT_BACKGROUND_OUTPUT_PATH,
        help="Where to save the calibrated background .npy file.",
    )
    parser.add_argument(
        "--background_preview_path",
        type=Path,
        default=DEFAULT_BACKGROUND_PREVIEW_PATH,
        help="Where to save the background preview image.",
    )
    parser.add_argument(
        "--roi_preview_path",
        type=Path,
        default=DEFAULT_ROI_PREVIEW_PATH,
        help="Where to save the ROI preview image.",
    )
    parser.add_argument(
        "--threshold_plot_path",
        type=Path,
        default=DEFAULT_THRESHOLD_PLOT_PATH,
        help="Where to save the threshold sweep plot.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to save the JSON report with calibration stats.",
    )
    parser.add_argument(
        "--disable_roi_refine",
        action="store_true",
        help="Disable local ROI refinement around the base ROI from the config.",
    )
    parser.add_argument("--roi_search_radius_x", type=int, default=24)
    parser.add_argument("--roi_search_radius_y", type=int, default=24)
    parser.add_argument("--roi_search_step", type=int, default=4)
    parser.add_argument("--threshold_min", type=int, default=4)
    parser.add_argument("--threshold_max", type=int, default=40)
    parser.add_argument("--threshold_step", type=int, default=1)
    parser.add_argument(
        "--empty_ratio_quantile",
        type=float,
        default=0.995,
        help="Quantile of empty-scene foreground ratio used during threshold selection.",
    )
    parser.add_argument(
        "--target_empty_ratio",
        type=float,
        default=0.005,
        help="Target upper bound for empty-scene foreground ratio at the selected diff_threshold.",
    )
    parser.add_argument(
        "--ratio_margin",
        type=float,
        default=0.002,
        help="Safety margin added above the empty-scene foreground-ratio quantile.",
    )
    parser.add_argument("--brightness_quantile_low", type=float, default=0.01)
    parser.add_argument("--brightness_quantile_high", type=float, default=0.99)
    parser.add_argument(
        "--brightness_margin",
        type=float,
        default=5.0,
        help="Extra brightness headroom added outside the selected quantile range.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_runtime_config = load_runtime_config(args.config)
    normal_image_paths = collect_image_paths(args.normal_dir)
    normal_frames = read_frames(normal_image_paths, max_frames=args.max_frames)

    anomaly_frames: list[np.ndarray] | None = None
    anomaly_image_paths: list[Path] | None = None
    if args.anomaly_dir is not None:
        anomaly_image_paths = collect_image_paths(args.anomaly_dir)
        anomaly_frames = read_frames(anomaly_image_paths)

    artifacts = autocalibrate_from_normals(
        normal_frames,
        base_runtime_config,
        refine_roi=not args.disable_roi_refine,
        roi_search_radius_x=args.roi_search_radius_x,
        roi_search_radius_y=args.roi_search_radius_y,
        roi_search_step=args.roi_search_step,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        empty_ratio_quantile=args.empty_ratio_quantile,
        target_empty_ratio=args.target_empty_ratio,
        ratio_margin=args.ratio_margin,
        brightness_quantile_low=args.brightness_quantile_low,
        brightness_quantile_high=args.brightness_quantile_high,
        brightness_margin=args.brightness_margin,
        anomaly_frames=anomaly_frames,
    )

    save_config_module(
        artifacts.detector_config,
        base_runtime_config.pipeline,
        args.output_config_path,
    )
    save_background(artifacts.background_reference, args.background_output_path)
    _, roi_small_stack, _ = prepare_blurred_stack(normal_frames, artifacts.detector_config)
    save_background_preview(
        roi_small_stack,
        artifacts.background_reference,
        args.background_preview_path,
    )
    base_roi_clamped = clamp_roi_to_frame(
        base_runtime_config.detector.roi,
        normal_frames[0].shape[1],
        normal_frames[0].shape[0],
    )
    save_roi_preview(
        normal_frames[0],
        base_roi_clamped,
        artifacts.selected_roi,
        args.roi_preview_path,
    )
    save_threshold_plot(
        artifacts.threshold_sweep,
        artifacts.detector_config.diff_threshold,
        artifacts.detector_config.min_foreground_ratio,
        args.threshold_plot_path,
    )

    report_payload = {
        "base_config": args.config,
        "normal_dir": str(args.normal_dir),
        "anomaly_dir": None if args.anomaly_dir is None else str(args.anomaly_dir),
        "normal_frames": len(normal_frames),
        "anomaly_frames": 0 if anomaly_frames is None else len(anomaly_frames),
        "base_roi": list(base_roi_clamped.as_xywh()),
        "selected_roi": list(artifacts.selected_roi.as_xywh()),
        "detector_config": {
            "roi": list(artifacts.detector_config.roi.as_xywh()),
            "processing_size": list(artifacts.detector_config.processing_size),
            "blur_kernel": artifacts.detector_config.blur_kernel,
            "diff_threshold": artifacts.detector_config.diff_threshold,
            "min_foreground_ratio": artifacts.detector_config.min_foreground_ratio,
            "enter_frames": artifacts.detector_config.enter_frames,
            "exit_frames": artifacts.detector_config.exit_frames,
            "morph_kernel_size": artifacts.detector_config.morph_kernel_size,
            "morph_open_iterations": artifacts.detector_config.morph_open_iterations,
            "morph_close_iterations": artifacts.detector_config.morph_close_iterations,
            "background_alpha": artifacts.detector_config.background_alpha,
            "use_adaptive_background": artifacts.detector_config.use_adaptive_background,
            "input_color_order": artifacts.detector_config.input_color_order,
            "min_brightness": artifacts.detector_config.min_brightness,
            "max_brightness": artifacts.detector_config.max_brightness,
        },
        "pipeline_config": {
            "classifier_roi": None
            if base_runtime_config.pipeline.classifier_roi is None
            else list(base_runtime_config.pipeline.classifier_roi.as_xywh()),
            "classifier_input_size": list(base_runtime_config.pipeline.classifier_input_size),
            "collect_frames_for_selection": base_runtime_config.pipeline.collect_frames_for_selection,
            "cooldown_frames": base_runtime_config.pipeline.cooldown_frames,
        },
        "roi_search_summary": artifacts.roi_search_summary,
        "normal_static_summary": artifacts.normal_static_summary,
        "anomaly_static_summary": artifacts.anomaly_static_summary,
        "threshold_sweep": serialize_threshold_sweep(artifacts.threshold_sweep),
        "outputs": {
            "config_path": str(args.output_config_path),
            "background_output_path": str(args.background_output_path),
            "background_preview_path": str(args.background_preview_path),
            "roi_preview_path": str(args.roi_preview_path),
            "threshold_plot_path": str(args.threshold_plot_path),
            "report_path": str(args.report_path),
        },
    }
    save_report(report_payload, args.report_path)

    print("=== Presence detector autocalibration ===")
    print(f"Normal frames used: {len(normal_frames)}")
    print(f"Base ROI: {base_roi_clamped.as_xywh()}")
    print(f"Selected ROI: {artifacts.selected_roi.as_xywh()}")
    print(
        "Brightness range: "
        f"{artifacts.detector_config.min_brightness:.1f} .. {artifacts.detector_config.max_brightness:.1f}"
    )
    print(f"Selected diff_threshold: {artifacts.detector_config.diff_threshold}")
    print(f"Selected min_foreground_ratio: {artifacts.detector_config.min_foreground_ratio:.6f}")
    print(
        "Normal signal rate after fit: "
        f"{artifacts.normal_static_summary['signal_rate']:.4f}"
    )
    if artifacts.anomaly_static_summary is not None:
        print(
            "Anomaly signal rate after fit: "
            f"{artifacts.anomaly_static_summary['signal_rate']:.4f}"
        )
    print(f"Saved config module to: {args.output_config_path}")
    print(f"Saved background reference to: {args.background_output_path}")
    print(f"Saved JSON report to: {args.report_path}")


if __name__ == "__main__":
    main()
