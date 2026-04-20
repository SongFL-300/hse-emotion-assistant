from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BlinkObservation:
    blink_event: bool
    eye_aspect_ratio: float
    blink_times: list[float]


class BlinkTracker:
    def __init__(self, *, close_threshold: float = 0.225, open_threshold: float = 0.265):
        self._close_threshold = float(close_threshold)
        self._open_threshold = float(open_threshold)
        self._blink_times: deque[float] = deque(maxlen=128)
        self._closed = False

    def reset(self) -> None:
        self._blink_times.clear()
        self._closed = False

    def observe(self, landmarks_px: list[tuple[int, int]], ts: float, *, frontal_score: float) -> BlinkObservation:
        ear = self._compute_ear(landmarks_px)
        blink_event = False
        if frontal_score < 0.45:
            return BlinkObservation(blink_event=False, eye_aspect_ratio=ear, blink_times=list(self._blink_times))
        if ear < self._close_threshold and not self._closed:
            self._closed = True
        elif ear > self._open_threshold and self._closed:
            self._closed = False
            blink_event = True
            self._blink_times.append(float(ts))
        return BlinkObservation(blink_event=blink_event, eye_aspect_ratio=ear, blink_times=list(self._blink_times))

    def _compute_ear(self, landmarks_px: list[tuple[int, int]]) -> float:
        left = self._single_eye_ear(landmarks_px, [33, 160, 158, 133, 153, 144])
        right = self._single_eye_ear(landmarks_px, [362, 385, 387, 263, 373, 380])
        values = [v for v in (left, right) if v > 0.0]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _single_eye_ear(self, landmarks_px: list[tuple[int, int]], idx: list[int]) -> float:
        try:
            p1, p2, p3, p4, p5, p6 = [landmarks_px[i] for i in idx]
        except Exception:
            return 0.0
        a = self._dist(p2, p6)
        b = self._dist(p3, p5)
        c = self._dist(p1, p4)
        if c <= 1e-6:
            return 0.0
        return float((a + b) / (2.0 * c))

    def _dist(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))
