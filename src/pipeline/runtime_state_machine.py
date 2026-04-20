from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import cv2
import numpy as np
import numpy.typing as npt

from configs.runtime.presence_config import RuntimePipelineConfig
from .frame_selector import BestFrameSelector, FrameCandidate
from .presence_detector import PresenceDetectionResult, PresenceDetector, crop_to_roi

UInt8Image = npt.NDArray[np.uint8]
ClassifierCallable = Callable[[UInt8Image], Any]


class PipelineState(str, Enum):
    EMPTY = "EMPTY"
    OBJECT_ENTERING = "OBJECT_ENTERING"
    OBJECT_PRESENT = "OBJECT_PRESENT"
    WAIT_UNTIL_EMPTY = "WAIT_UNTIL_EMPTY"


@dataclass(slots=True)
class RuntimePipelineResult:
    state: PipelineState
    presence: PresenceDetectionResult
    should_classify: bool
    # Python desktop/debug layer only. The future ESP runtime should expose a
    # smaller event-oriented result instead of raw classifier/debug payloads.
    classifier_input: UInt8Image | None = None
    classification: Any | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class RuntimePipeline:
    def __init__(
            self,
            presence_detector: PresenceDetector,
            frame_selector: BestFrameSelector,
            classifier: ClassifierCallable | None,
            config: RuntimePipelineConfig,
    ) -> None:
        self.presence_detector = presence_detector
        self.frame_selector = frame_selector
        self.classifier = classifier
        self.config = config

        self.state = PipelineState.EMPTY
        self._frame_index = 0
        self._cooldown_remaining = 0
        self._object_classified = False

    def reset(self, reset_background: bool = False) -> None:
        self.presence_detector.reset(reset_background=reset_background)
        self.frame_selector.clear()
        self.state = PipelineState.EMPTY
        self._frame_index = 0
        self._cooldown_remaining = 0
        self._object_classified = False

    def process_frame(self, frame: UInt8Image) -> RuntimePipelineResult:
        self._frame_index += 1
        previous_state = self.state
        presence = self.presence_detector.update(frame)
        selected_candidate = self._advance(frame, presence)

        classifier_input: UInt8Image | None = None
        classification: Any | None = None
        should_classify = selected_candidate is not None

        if selected_candidate is not None:
            classifier_input = self._prepare_classifier_input(selected_candidate.frame)
            if self.classifier is not None:
                classification = self.classifier(np.array(classifier_input, copy=True))

        return RuntimePipelineResult(
            state=self.state,
            presence=presence,
            should_classify=should_classify,
            classifier_input=classifier_input,
            classification=classification,
            debug=self._build_debug(previous_state, selected_candidate),
        )

    def _advance(
            self,
            frame: UInt8Image,
            presence: PresenceDetectionResult,
    ) -> FrameCandidate | None:
        if self.state == PipelineState.EMPTY:
            return self._handle_empty(frame, presence)
        if self.state == PipelineState.OBJECT_ENTERING:
            return self._handle_object_entering(frame, presence)
        if self.state == PipelineState.OBJECT_PRESENT:
            return self._handle_object_present(frame, presence)
        if self.state == PipelineState.WAIT_UNTIL_EMPTY:
            return self._handle_wait_until_empty(presence)

        raise RuntimeError(f"Unsupported pipeline state: {self.state}")

    def _handle_empty(
            self,
            frame: UInt8Image,
            presence: PresenceDetectionResult,
    ) -> FrameCandidate | None:
        self._clear_object()

        if not (presence.signal_active or presence.is_present):
            return None

        self._collect(frame, presence)
        if presence.is_present:
            self.state = PipelineState.OBJECT_PRESENT
        else:
            self.state = PipelineState.OBJECT_ENTERING

        return None

    def _handle_object_entering(
            self,
            frame: UInt8Image,
            presence: PresenceDetectionResult,
    ) -> FrameCandidate | None:
        if not (presence.signal_active or presence.is_present):
            self._clear_object()
            self.state = PipelineState.EMPTY
            return None

        self._collect(frame, presence)
        if presence.is_present:
            self.state = PipelineState.OBJECT_PRESENT

        return None

    def _handle_object_present(
            self,
            frame: UInt8Image,
            presence: PresenceDetectionResult,
    ) -> FrameCandidate | None:
        if not (presence.signal_active or presence.is_present):
            self._clear_object()
            self.state = PipelineState.EMPTY
            return None

        self._collect(frame, presence)
        if self._object_classified:
            return None

        if len(self.frame_selector) < self.config.collect_frames_for_selection:
            return None

        selected_candidate = self.frame_selector.get_best_candidate()
        if selected_candidate is None:
            return None

        self._object_classified = True
        self._cooldown_remaining = self.config.cooldown_frames
        self.state = PipelineState.WAIT_UNTIL_EMPTY
        return selected_candidate

    def _handle_wait_until_empty(
            self,
            presence: PresenceDetectionResult,
    ) -> FrameCandidate | None:
        if presence.signal_active or presence.is_present:
            self._cooldown_remaining = self.config.cooldown_frames
            return None

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None

        self._clear_object()
        self.state = PipelineState.EMPTY
        return None

    def _collect(self, frame: UInt8Image, presence: PresenceDetectionResult) -> None:
        self.frame_selector.add(
            frame=frame,
            score=presence.foreground_ratio,
            meta={
                "frame_index": self._frame_index,
                "foreground_ratio": presence.foreground_ratio,
                "brightness": presence.brightness,
            },
        )

    def _clear_object(self) -> None:
        self.frame_selector.clear()
        self._object_classified = False
        self._cooldown_remaining = 0

    def _prepare_classifier_input(self, frame: UInt8Image) -> UInt8Image:
        roi = self.config.classifier_roi or self.presence_detector.config.roi
        cropped = crop_to_roi(frame, roi)
        resized = cv2.resize(
            cropped,
            self.config.classifier_input_size,
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(resized)

    def _build_debug(
            self,
            previous_state: PipelineState,
            selected_candidate: FrameCandidate | None,
    ) -> dict[str, Any]:
        # Desktop replay/calibration payload only. The embedded runtime should
        # not mirror selector scores or debug metadata as part of its API.
        return {
            "frame_index": self._frame_index,
            "state_changed": previous_state != self.state,
            "cooldown_remaining": self._cooldown_remaining,
            "selector_scores": self.frame_selector.scores(),
            "selected_score": None if selected_candidate is None else selected_candidate.score,
            "selected_meta": {} if selected_candidate is None else selected_candidate.meta,
        }
