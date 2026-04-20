from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import time
from typing import Deque


STANDARD_EMOTIONS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
    "unknown",
)


def normalize_emotion_label(label: str) -> str:
    # 兼容 mini_XCEPTION / DeepFace 的命名差异
    mapping = {
        "scared": "fear",
        "fearful": "fear",
        "surprised": "surprise",
    }
    normalized = mapping.get(label.lower().strip(), label.lower().strip())
    if normalized not in STANDARD_EMOTIONS:
        return "unknown"
    return normalized


@dataclass(frozen=True)
class EmotionSnapshot:
    ts: float
    emotion: str
    probability: float
    scores: dict[str, float] | None = None
    face_detected: bool = True
    valence: float = 0.0
    arousal: float = 0.0
    uncertain: bool = False
    pose_yaw: float = 0.0
    pose_pitch: float = 0.0
    pose_roll: float = 0.0
    frontal_score: float = 1.0


def emotion_distance(a: EmotionSnapshot, b: EmotionSnapshot) -> float:
    """
    A lightweight “emotion change” metric in [0, 1].

    - If scores are available, use mean absolute difference over STANDARD_EMOTIONS.
    - Otherwise, fallback to (label change ? 1 : abs(probability delta)).
    """
    if not a.face_detected or not b.face_detected:
        return 0.0

    if a.scores and b.scores:
        va = [float(a.scores.get(k, 0.0)) for k in STANDARD_EMOTIONS]
        vb = [float(b.scores.get(k, 0.0)) for k in STANDARD_EMOTIONS]
        diff = sum(abs(x - y) for x, y in zip(va, vb, strict=False)) / max(1, len(STANDARD_EMOTIONS))
        return max(0.0, min(1.0, float(diff)))

    affect_diff = (abs(float(a.valence) - float(b.valence)) + abs(float(a.arousal) - float(b.arousal))) / 2.0
    if affect_diff > 0.0:
        return max(0.0, min(1.0, float(affect_diff)))

    if a.emotion != b.emotion:
        return 1.0
    return max(0.0, min(1.0, abs(float(b.probability) - float(a.probability))))


class EmotionTimeline:
    """
    保存最近一段时间的情绪快照，并给出“当前情绪 + 变化趋势”的可读摘要。
    """

    def __init__(self, *, window_seconds: float = 8.0):
        self._window_seconds = float(window_seconds)
        self._items: Deque[EmotionSnapshot] = deque()

    def add(self, snap: EmotionSnapshot) -> None:
        self._items.append(snap)
        self._gc()

    def latest(self) -> EmotionSnapshot | None:
        return self._items[-1] if self._items else None

    def summary(self) -> str | None:
        self._gc()
        if not self._items:
            return None

        last = self._items[-1]
        if not last.face_detected:
            return "未检测到人脸（可能离开镜头或遮挡）。"

        affect_line = (
            f"valence={last.valence:+.2f}({self._valence_band(last.valence)})"
            f"；arousal={last.arousal:.2f}({self._arousal_band(last.arousal)})"
            f"；confidence={last.probability:.2f}"
        )
        pose_line = f"；frontal={last.frontal_score:.2f}"

        dominant_line = ""
        if last.emotion == "unknown" or last.uncertain:
            dominant_line = "；dominant=unknown"
        else:
            dominant_line = f"；dominant={last.emotion}"

        top3_str = ""
        if last.scores:
            items = sorted(last.scores.items(), key=lambda kv: kv[1], reverse=True)
            top3 = items[:3]
            top3_str = "；top3=" + ", ".join([f"{k}:{v:.2f}" for k, v in top3])

        first = self._items[0]
        trend_line = ""
        valence_delta = float(last.valence - first.valence)
        arousal_delta = float(last.arousal - first.arousal)
        valence_trend = "更积极" if valence_delta > 0.12 else "更收敛" if valence_delta < -0.12 else "平稳"
        arousal_trend = "更激活" if arousal_delta > 0.12 else "更平静" if arousal_delta < -0.12 else "平稳"
        trend_line = f"；趋势=valence:{valence_trend}({valence_delta:+.2f}), arousal:{arousal_trend}({arousal_delta:+.2f})"

        volatility = self._volatility_level()
        vol_line = f"；波动={volatility}" if volatility else ""

        return f"当前={affect_line}{pose_line}{dominant_line}{top3_str}{trend_line}{vol_line}"

    def structured_signal(self) -> str | None:
        self._gc()
        if not self._items:
            return None
        last = self._items[-1]
        if not last.face_detected:
            return "[local_visual_signal]\nface_detected=false\n[/local_visual_signal]"
        dominant = "unknown" if last.uncertain else last.emotion
        trend = self._trend_compact()
        return (
            "[local_visual_signal]\n"
            f"face_detected=true\n"
            f"dominant={dominant}\n"
            f"confidence={last.probability:.2f}\n"
            f"valence={last.valence:+.2f}\n"
            f"arousal={last.arousal:.2f}\n"
            f"frontal_score={last.frontal_score:.2f}\n"
            f"trend={trend}\n"
            "[/local_visual_signal]"
        )

    def change_score(self) -> float | None:
        """
        Return the emotion change score (see emotion_distance) between the first and last
        snapshot within the current time window.
        """
        self._gc()
        if len(self._items) < 2:
            return None
        return emotion_distance(self._items[0], self._items[-1])

    def _gc(self) -> None:
        now = time.time()
        cutoff = now - self._window_seconds
        while self._items and self._items[0].ts < cutoff:
            self._items.popleft()

    def _volatility_level(self) -> str | None:
        if len(self._items) < 6:
            return None
        values = [abs(s.valence) * 0.6 + abs(s.arousal - 0.25) * 0.4 for s in self._items if s.face_detected]
        if len(values) < 6:
            return None
        mean = sum(values) / len(values)
        var = sum((p - mean) ** 2 for p in values) / max(1, len(values) - 1)
        std = math.sqrt(var)
        if std < 0.06:
            return "低"
        if std < 0.14:
            return "中"
        return "高"

    def _trend_compact(self) -> str:
        if len(self._items) < 2:
            return "stable"
        first = self._items[0]
        last = self._items[-1]
        valence_delta = float(last.valence - first.valence)
        arousal_delta = float(last.arousal - first.arousal)
        if abs(valence_delta) < 0.10 and abs(arousal_delta) < 0.10:
            return "stable"
        if arousal_delta > 0.12:
            return "more_activated"
        if arousal_delta < -0.12:
            return "calmer"
        if valence_delta > 0.12:
            return "more_positive"
        if valence_delta < -0.12:
            return "more_negative"
        return "mixed"

    def _valence_band(self, value: float) -> str:
        if value > 0.22:
            return "偏正向"
        if value < -0.22:
            return "偏负向"
        return "中性"

    def _arousal_band(self, value: float) -> str:
        if value > 0.62:
            return "高"
        if value > 0.35:
            return "中"
        return "低"
