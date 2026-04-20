from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

UInt8Image = npt.NDArray[np.uint8]


@dataclass(slots=True)
class FrameCandidate:
    frame: UInt8Image
    score: float
    # Desktop-only debug metadata. Do not rely on this as an embedded-facing
    # interface.
    meta: dict[str, Any] = field(default_factory=dict)


class BestFrameSelector:
    def __init__(self, max_frames: int = 3) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")

        self.max_frames = max_frames
        self._candidates: list[FrameCandidate] = []

    def add(
            self,
            frame: UInt8Image,
            score: float,
            meta: dict[str, Any] | None = None,
    ) -> None:
        candidate = FrameCandidate(
            frame=np.array(frame, copy=True),
            score=float(score),
            meta=meta or {},
        )
        self._candidates.append(candidate)
        if len(self._candidates) > self.max_frames:
            self._candidates.pop(0)

    def get_best(self) -> UInt8Image | None:
        candidate = self.get_best_candidate()
        if candidate is None:
            return None
        return np.array(candidate.frame, copy=True)

    def get_best_candidate(self) -> FrameCandidate | None:
        if not self._candidates:
            return None
        return max(self._candidates, key=lambda candidate: candidate.score)

    def clear(self) -> None:
        self._candidates.clear()

    def scores(self) -> list[float]:
        return [candidate.score for candidate in self._candidates]

    def __len__(self) -> int:
        return len(self._candidates)
