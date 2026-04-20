from __future__ import annotations

# Desktop-only replay/debug utility. Keep this on the laptop side; it is not a
# target for the future ESP port.

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from configs.runtime.presence_config import RuntimeConfig
from src.pipeline.frame_selector import BestFrameSelector
from src.pipeline.presence_detector import PresenceDetector
from src.pipeline.runtime_state_machine import PipelineState, RuntimePipeline

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class NamedFrame:
    index: int
    name: str
    image: np.ndarray


class DebugClassifier:
    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, frame: np.ndarray) -> dict[str, Any]:
        self.call_count += 1
        return {
            "call_index": self.call_count,
            "shape": list(frame.shape),
            "mean_intensity": float(np.mean(frame)),
        }


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


def load_frames_from_directory(input_dir: Path) -> list[NamedFrame]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not image_paths:
        raise ValueError(f"No image files found in: {input_dir}")

    frames: list[NamedFrame] = []
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        frames.append(NamedFrame(index=index, name=image_path.name, image=image))
    return frames


def load_frames_from_video(video_path: Path) -> list[NamedFrame]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: list[NamedFrame] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(NamedFrame(index=index, name=f"{video_path.stem}_{index:06d}.png", image=frame))
        index += 1

    capture.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return frames


def create_overlay(
    frame: np.ndarray,
    pipeline_state: PipelineState,
    foreground_ratio: float,
    brightness: float,
    brightness_ok: bool,
    roi: tuple[int, int, int, int],
    mask: np.ndarray,
    diff: np.ndarray,
) -> np.ndarray:
    overlay = np.array(frame, copy=True)
    x, y, width, height = roi
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)

    state_text = f"state={pipeline_state.value}"
    ratio_text = f"fg_ratio={foreground_ratio:.4f}"
    brightness_text = f"brightness={brightness:.1f} ok={int(brightness_ok)}"

    cv2.putText(overlay, state_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(overlay, ratio_text, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(overlay, brightness_text, (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    mask_panel = cv2.cvtColor(
        cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )
    diff_panel = cv2.cvtColor(
        cv2.resize(diff, (width, height), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )

    top = y
    left_mask = max(0, overlay.shape[1] - (2 * width + 24))
    left_diff = max(0, overlay.shape[1] - (width + 12))

    bottom = min(overlay.shape[0], top + height)
    right_mask = min(overlay.shape[1], left_mask + width)
    right_diff = min(overlay.shape[1], left_diff + width)

    mask_panel = mask_panel[:bottom - top, :right_mask - left_mask]
    diff_panel = diff_panel[:bottom - top, :right_diff - left_diff]

    overlay[top:bottom, left_mask:right_mask] = mask_panel
    overlay[top:bottom, left_diff:right_diff] = diff_panel

    cv2.putText(overlay, "mask", (left_mask, max(20, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, "diff", (left_diff, max(20, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return overlay


def save_classifier_frame(save_dir: Path, frame_index: int, classifier_frame: np.ndarray) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"classifier_frame_{frame_index:06d}.png"
    if not cv2.imwrite(str(output_path), classifier_frame):
        raise ValueError(f"Could not save classifier frame to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="presence_config",
        help="Runtime config module name from configs/runtime/ (without .py)",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=Path, help="Directory with ordered debug frames")
    input_group.add_argument("--video_path", type=Path, help="Video file for local replay")
    parser.add_argument(
        "--background_path",
        type=Path,
        default=None,
        help="Path to a saved background reference",
    )
    parser.add_argument(
        "--bootstrap_empty_frames",
        type=int,
        default=0,
        help="How many initial empty frames to use when background_path is not provided",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Optional directory for debug overlays and classifier crops",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_runtime_config(args.config)

    if args.input_dir is not None:
        named_frames = load_frames_from_directory(args.input_dir)
    else:
        named_frames = load_frames_from_video(args.video_path)

    detector = PresenceDetector(config.detector)
    if args.background_path is not None:
        detector.load_background(args.background_path)
    else:
        if args.bootstrap_empty_frames <= 0:
            raise ValueError(
                "Either --background_path must be provided or --bootstrap_empty_frames must be positive."
            )
        bootstrap_frames = [item.image for item in named_frames[:args.bootstrap_empty_frames]]
        if len(bootstrap_frames) < args.bootstrap_empty_frames:
            raise ValueError("Not enough frames to bootstrap the background.")
        detector.fit_background(bootstrap_frames)

    classifier = DebugClassifier()
    selector = BestFrameSelector(max_frames=config.pipeline.collect_frames_for_selection)
    pipeline = RuntimePipeline(
        presence_detector=detector,
        frame_selector=selector,
        classifier=classifier,
        config=config.pipeline,
    )
    pipeline.reset(reset_background=False)

    roi = config.detector.roi.as_xywh()
    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("=== Replay started ===")
    print(f"Total frames: {len(named_frames)}")
    if args.background_path is not None:
        print(f"Background loaded from: {args.background_path}")
    else:
        print(f"Background bootstrapped from first {args.bootstrap_empty_frames} frames")

    for named_frame in named_frames:
        result = pipeline.process_frame(named_frame.image)

        if result.debug["state_changed"] or result.should_classify:
            print(
                f"[frame {named_frame.index:05d}] "
                f"{named_frame.name} "
                f"state={result.state.value} "
                f"fg_ratio={result.presence.foreground_ratio:.4f} "
                f"selector_scores={result.debug['selector_scores']}"
            )

        if result.should_classify:
            selected_score = result.debug["selected_score"]
            print(
                f"  classification_triggered "
                f"best_score={selected_score:.4f} "
                f"meta={result.debug['selected_meta']} "
                f"result={result.classification}"
            )
            if save_dir is not None and result.classifier_input is not None:
                save_classifier_frame(save_dir, named_frame.index, result.classifier_input)

        if save_dir is not None:
            overlay = create_overlay(
                frame=named_frame.image,
                pipeline_state=result.state,
                foreground_ratio=result.presence.foreground_ratio,
                brightness=result.presence.brightness,
                brightness_ok=result.presence.brightness_ok,
                roi=roi,
                mask=result.presence.mask,
                diff=result.presence.diff,
            )
            output_path = save_dir / f"{named_frame.index:06d}.png"
            if not cv2.imwrite(str(output_path), overlay):
                raise ValueError(f"Could not save debug overlay to: {output_path}")

    print("=== Replay finished ===")
    print(f"Classifier calls: {classifier.call_count}")
    print(f"Final state: {pipeline.state.value}")


if __name__ == "__main__":
    main()
