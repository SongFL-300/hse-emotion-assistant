from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..emotion.structured import RMESSpotEvent


@dataclass(frozen=True)
class RMESExperimentMetrics:
    total_events: int = 0
    accepted_events: int = 0
    average_score: float = 0.0
    acceptance_ratio: float = 0.0


class RMESAdapter:
    def __init__(
        self,
        *,
        window_size: int = 18,
        cooldown_s: float = 1.2,
        min_score: float = 0.58,
        acceptance_score: float = 0.72,
    ) -> None:
        self._window_size = max(8, int(window_size))
        self._cooldown_s = float(cooldown_s)
        self._min_score = float(min_score)
        self._acceptance_score = float(acceptance_score)
        self._history: deque[dict[str, Any]] = deque(maxlen=self._window_size)
        self._events: deque[RMESSpotEvent] = deque(maxlen=64)
        self._accepted_events: deque[RMESSpotEvent] = deque(maxlen=32)
        self._last_emit_ts = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._events.clear()
        self._accepted_events.clear()
        self._last_emit_ts = 0.0

    def observe(
        self,
        *,
        ts: float,
        window_id: int,
        subtle_metrics: dict[str, float],
        pose: dict[str, float],
        dominant_emotion: str,
        valence: float,
        arousal: float,
        confidence: float,
        au_intensities: dict[str, float] | None,
    ) -> RMESSpotEvent | None:
        item = {
            "ts": float(ts),
            "metrics": {str(k): float(v) for k, v in subtle_metrics.items()},
            "frontal_score": float(pose.get("frontal_score", 0.0)),
            "dominant_emotion": str(dominant_emotion),
            "valence": float(valence),
            "arousal": float(arousal),
            "confidence": float(confidence),
            "au_intensities": {str(k): float(v) for k, v in (au_intensities or {}).items()},
        }
        self._history.append(item)
        if len(self._history) < self._window_size:
            return None

        score = self._compute_score()
        peak = max(self._history, key=lambda row: self._frame_energy(row["metrics"]))
        frontal_mean = sum(row["frontal_score"] for row in self._history) / len(self._history)
        conf_mean = sum(row["confidence"] for row in self._history) / len(self._history)
        quality_gate = frontal_mean >= 0.58 and conf_mean >= 0.38
        if score < self._min_score or not quality_gate:
            return None
        if float(ts) - self._last_emit_ts < self._cooldown_s:
            return None

        self._last_emit_ts = float(ts)
        first = self._history[0]
        last = self._history[-1]
        event = RMESSpotEvent(
            window_id=window_id,
            clip_start_ts=float(first["ts"]),
            clip_end_ts=float(last["ts"]),
            peak_ts=float(peak["ts"]),
            spot_score=float(score),
            confidence=float(min(0.95, score * 0.92)),
            quality_gate_passed=quality_gate,
            interpretation=self._build_interpretation(),
            valence_delta=float(last["valence"] - first["valence"]),
            arousal_delta=float(last["arousal"] - first["arousal"]),
            dominant_emotion=str(peak["dominant_emotion"]),
        )
        self._events.append(event)
        if score >= self._acceptance_score and quality_gate:
            self._accepted_events.append(event)
        return event

    def get_events(self) -> list[RMESSpotEvent]:
        return list(self._events)

    def get_recent_events_for_window(self, window_id: int) -> list[RMESSpotEvent]:
        return [event for event in self._events if int(event.window_id) == int(window_id)]

    def get_accepted_events_for_window(self, window_id: int) -> list[RMESSpotEvent]:
        return [event for event in self._accepted_events if int(event.window_id) == int(window_id)]

    def get_metrics(self) -> RMESExperimentMetrics:
        total = len(self._events)
        accepted = len(self._accepted_events)
        avg = sum(event.spot_score for event in self._events) / total if total else 0.0
        ratio = float(accepted / total) if total else 0.0
        return RMESExperimentMetrics(total_events=total, accepted_events=accepted, average_score=avg, acceptance_ratio=ratio)

    def is_accepted_for_mainline(self) -> bool:
        metrics = self.get_metrics()
        return metrics.total_events >= 3 and metrics.acceptance_ratio >= 0.55 and metrics.average_score >= self._acceptance_score

    def _compute_score(self) -> float:
        if len(self._history) < 3:
            return 0.0
        energies = [self._frame_energy(row["metrics"]) for row in self._history]
        frontal = [row["frontal_score"] for row in self._history]
        conf = [row["confidence"] for row in self._history]
        valences = [row["valence"] for row in self._history]
        arousals = [row["arousal"] for row in self._history]
        motion = sum(abs(b - a) for a, b in zip(energies[:-1], energies[1:], strict=False)) / max(1, len(energies) - 1)
        focus = max(energies) - min(energies)
        affect_shift = (abs(valences[-1] - valences[0]) + abs(arousals[-1] - arousals[0])) / 2.0
        quality = (sum(frontal) / len(frontal)) * 0.55 + (sum(conf) / len(conf)) * 0.45
        raw = motion * 0.42 + focus * 0.28 + affect_shift * 0.18 + quality * 0.12
        return float(max(0.0, min(0.99, raw)))

    def _frame_energy(self, metrics: dict[str, float]) -> float:
        weights = {
            "eye_open": 1.0,
            "eye_asymmetry": 1.1,
            "mouth_open": 0.9,
            "brow_raise": 1.0,
            "brow_asymmetry": 1.15,
            "mouth_corner_tilt": 1.0,
            "lip_press": 1.05,
        }
        total = 0.0
        for key, value in metrics.items():
            total += abs(float(value)) * float(weights.get(key, 1.0))
        return total

    def _build_interpretation(self) -> str:
        if not self._history:
            return ""
        recent = self._history[-1]["metrics"]
        ranked = sorted(((k, abs(v)) for k, v in recent.items()), key=lambda item: item[1], reverse=True)[:3]
        if not ranked:
            return "局部细微变化较弱。"
        names = ", ".join(f"{name}:{score:.3f}" for name, score in ranked)
        return f"RMES 实验分支检测到短时局部变化峰值，主要来自 {names}。"
