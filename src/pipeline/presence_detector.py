from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from configs.runtime.presence_config import PresenceDetectorConfig, ROIConfig

UInt8Image = npt.NDArray[np.uint8]
Float32Image = npt.NDArray[np.float32]


def crop_to_roi(frame: UInt8Image, roi: ROIConfig) -> UInt8Image:
    if frame.ndim not in {2, 3}:
        raise ValueError("Frame must be either grayscale or color.")

    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = roi.as_xywh()
    if x + width > frame_width or y + height > frame_height:
        raise ValueError(
            "ROI is outside of frame bounds. "
            f"Frame size: {(frame_width, frame_height)}, ROI: {(x, y, width, height)}."
        )

    return np.ascontiguousarray(frame[y:y + height, x:x + width])


@dataclass(slots=True)
class PreparedFrame:
    roi_color: UInt8Image
    roi_small: UInt8Image
    roi_blurred: UInt8Image


@dataclass(slots=True)
class PresenceDetectionResult:
    is_present: bool
    signal_active: bool
    foreground_ratio: float
    entered: bool
    exited: bool
    brightness: float
    brightness_ok: bool
    # Desktop-only debug artifacts. Keep them for calibration/replay, but do
    # not treat them as part of the future embedded runtime API.
    mask: UInt8Image
    diff: UInt8Image
    roi_small: UInt8Image


class FramePreprocessor:
    def __init__(self, config: PresenceDetectorConfig) -> None:
        self.config = config

    def prepare(self, frame: UInt8Image) -> PreparedFrame:
        roi_color = crop_to_roi(frame, self.config.roi)
        gray = self._to_grayscale(roi_color)
        roi_small = cv2.resize(
            gray,
            self.config.processing_size,
            interpolation=cv2.INTER_AREA,
        )
        roi_blurred = cv2.GaussianBlur(
            roi_small,
            (self.config.blur_kernel, self.config.blur_kernel),
            0,
        )
        return PreparedFrame(
            roi_color=roi_color,
            roi_small=roi_small,
            roi_blurred=roi_blurred,
        )

    def _to_grayscale(self, frame: UInt8Image) -> UInt8Image:
        if frame.ndim == 2:
            return frame.astype(np.uint8, copy=False)

        if self.config.input_color_order.lower() == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class BackgroundModel:
    def __init__(self, alpha: float, processing_size: tuple[int, int]) -> None:
        self.alpha = alpha
        self.processing_size = processing_size
        self._reference: Float32Image | None = None

    @property
    def is_ready(self) -> bool:
        return self._reference is not None

    @property
    def image(self) -> UInt8Image | None:
        if self._reference is None:
            return None
        return np.clip(np.rint(self._reference), 0, 255).astype(np.uint8)

    def clear(self) -> None:
        self._reference = None

    def fit(self, frames: Sequence[PreparedFrame]) -> None:
        if not frames:
            raise ValueError("At least one frame is required to fit the background.")

        blurred_frames = [
            prepared_frame.roi_blurred.astype(np.float32)
            for prepared_frame in frames
        ]
        self._reference = np.mean(blurred_frames, axis=0, dtype=np.float32)

    def set_reference(
            self,
            background_reference: npt.ArrayLike,
            processor: FramePreprocessor,
    ) -> None:
        background_array = np.asarray(background_reference)

        if background_array.ndim == 3:
            processed = processor.prepare(background_array.astype(np.uint8))
            self._reference = processed.roi_blurred.astype(np.float32)
            return

        if background_array.ndim != 2:
            raise ValueError("Background reference must be a processed image or a raw frame.")

        expected_shape = (
            self.processing_size[1],
            self.processing_size[0],
        )
        if background_array.shape != expected_shape:
            raise ValueError(
                "Background reference shape does not match processing_size. "
                f"Expected {expected_shape}, got {background_array.shape}."
            )

        self._reference = background_array.astype(np.float32)

    def load(self, path: str | Path, processor: FramePreprocessor) -> None:
        # Desktop convenience helper. For ESP, use a pre-calibrated small
        # background buffer instead of generic file/image loading.
        background_path = Path(path)
        if not background_path.exists():
            raise FileNotFoundError(f"Background file not found: {background_path}")

        if background_path.suffix.lower() == ".npy":
            self.set_reference(np.load(background_path), processor)
            return

        image = cv2.imread(str(background_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read background image: {background_path}")
        self.set_reference(image, processor)

    def difference(self, current_frame: UInt8Image) -> UInt8Image:
        reference = self.image
        if reference is None:
            raise RuntimeError("Background is not ready.")
        return cv2.absdiff(current_frame, reference)

    def update(self, current_frame: UInt8Image) -> None:
        if self._reference is None:
            self._reference = current_frame.astype(np.float32)
            return

        self._reference = (
                (1.0 - self.alpha) * self._reference
                + self.alpha * current_frame.astype(np.float32)
        )


class PresenceGate:
    def __init__(self, enter_frames: int, exit_frames: int) -> None:
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self._is_present = False
        self._enter_counter = 0
        self._exit_counter = 0

    def reset(self) -> None:
        self._is_present = False
        self._enter_counter = 0
        self._exit_counter = 0

    def update(self, signal_active: bool) -> tuple[bool, bool, bool]:
        entered = False
        exited = False

        if self._is_present:
            if signal_active:
                self._exit_counter = 0
            else:
                self._exit_counter += 1
                if self._exit_counter >= self.exit_frames:
                    self._is_present = False
                    self._enter_counter = 0
                    self._exit_counter = 0
                    exited = True
        else:
            if signal_active:
                self._enter_counter += 1
                if self._enter_counter >= self.enter_frames:
                    self._is_present = True
                    self._enter_counter = 0
                    entered = True
            else:
                self._enter_counter = 0
                self._exit_counter = 0

        return self._is_present, entered, exited


class PresenceDetector:
    def __init__(
            self,
            config: PresenceDetectorConfig,
            background_reference: npt.ArrayLike | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = FramePreprocessor(config)
        self.background = BackgroundModel(
            alpha=config.background_alpha,
            processing_size=config.processing_size,
        )
        self.gate = PresenceGate(
            enter_frames=config.enter_frames,
            exit_frames=config.exit_frames,
        )
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )

        if background_reference is not None:
            self.set_background_reference(background_reference)

    @property
    def has_background(self) -> bool:
        return self.background.is_ready

    @property
    def background_reference(self) -> UInt8Image | None:
        return self.background.image

    def reset(self, reset_background: bool = False) -> None:
        self.gate.reset()
        if reset_background:
            self.background.clear()

    def preprocess(self, frame: UInt8Image) -> PreparedFrame:
        return self.preprocessor.prepare(frame)

    def fit_background(self, frames: Sequence[UInt8Image]) -> None:
        prepared_frames = [self.preprocess(frame) for frame in frames]
        self.background.fit(prepared_frames)
        self.gate.reset()

    def set_background_reference(self, background_reference: npt.ArrayLike) -> None:
        self.background.set_reference(background_reference, self.preprocessor)
        self.gate.reset()

    def load_background(self, path: str | Path) -> None:
        self.background.load(path, self.preprocessor)
        self.gate.reset()

    def build_mask(self, diff: UInt8Image) -> UInt8Image:
        _, mask = cv2.threshold(
            diff,
            self.config.diff_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        if self.config.morph_open_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self._morph_kernel,
                iterations=self.config.morph_open_iterations,
            )
        if self.config.morph_close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self._morph_kernel,
                iterations=self.config.morph_close_iterations,
            )

        return mask

    def update(self, frame: UInt8Image) -> PresenceDetectionResult:
        prepared = self.preprocess(frame)
        brightness = float(np.mean(prepared.roi_blurred))
        brightness_ok = self._is_brightness_ok(brightness)

        if not self.has_background:
            empty_image = np.zeros_like(prepared.roi_blurred, dtype=np.uint8)
            return PresenceDetectionResult(
                is_present=False,
                signal_active=False,
                foreground_ratio=0.0,
                entered=False,
                exited=False,
                brightness=brightness,
                brightness_ok=brightness_ok,
                mask=empty_image,
                diff=empty_image.copy(),
                roi_small=prepared.roi_small,
            )

        diff = self.background.difference(prepared.roi_blurred)
        mask = self.build_mask(diff)
        foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        signal_active = brightness_ok and foreground_ratio >= self.config.min_foreground_ratio
        is_present, entered, exited = self.gate.update(signal_active)

        if (
                self.config.use_adaptive_background
                and not is_present
                and not signal_active
                and brightness_ok
        ):
            self.background.update(prepared.roi_blurred)

        return PresenceDetectionResult(
            is_present=is_present,
            signal_active=signal_active,
            foreground_ratio=foreground_ratio,
            entered=entered,
            exited=exited,
            brightness=brightness,
            brightness_ok=brightness_ok,
            mask=mask,
            diff=diff,
            roi_small=prepared.roi_small,
        )

    def _is_brightness_ok(self, brightness: float) -> bool:
        return self.config.min_brightness <= brightness <= self.config.max_brightness
