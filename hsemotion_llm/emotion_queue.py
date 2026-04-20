from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import time
from typing import Deque

from .emotion.state import EmotionSnapshot, emotion_distance


def _shorten(s: str, *, max_chars: int = 48) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"


@dataclass(frozen=True)
class EmotionCue:
    ts: float
    stage: str
    role: str
    text_excerpt: str
    snapshot: EmotionSnapshot | None
    summary: str | None


@dataclass(frozen=True)
class EmotionShift:
    ts: float
    stage: str
    role: str
    text_excerpt: str
    score: float
    from_emotion: str | None
    to_emotion: str | None
    from_signal: str
    to_signal: str
    from_summary: str | None
    to_summary: str | None


class EmotionQueue:
    """
    A lightweight local queue that tracks notable emotion changes around dialogue turns.

    It is intentionally conservative:
    - Only records "notable shifts" above a threshold to keep prompt small.
    - Designed to feed the LLM as internal context (not shown to the user by default).
    """

    def __init__(self, *, shift_threshold: float = 0.22, max_cues: int = 80, max_shifts: int = 24):
        self._shift_threshold = float(shift_threshold)
        self._cues: Deque[EmotionCue] = deque(maxlen=int(max_cues))
        self._shifts: Deque[EmotionShift] = deque(maxlen=int(max_shifts))

    @property
    def shift_threshold(self) -> float:
        return self._shift_threshold

    @shift_threshold.setter
    def shift_threshold(self, v: float) -> None:
        self._shift_threshold = float(v)

    def record(
        self,
        *,
        stage: str,
        role: str,
        text: str | None,
        snapshot: EmotionSnapshot | None,
        summary: str | None,
        ts: float | None = None,
    ) -> None:
        now = float(ts if ts is not None else time.time())
        cue = EmotionCue(
            ts=now,
            stage=str(stage),
            role=str(role),
            text_excerpt=_shorten(text or ""),
            snapshot=snapshot,
            summary=summary.strip() if summary else None,
        )

        prev = self._cues[-1] if self._cues else None
        self._cues.append(cue)

        if not prev or not prev.snapshot or not cue.snapshot:
            return

        score = float(emotion_distance(prev.snapshot, cue.snapshot))
        if score < self._shift_threshold:
            return

        from_em = prev.snapshot.emotion if prev.snapshot else None
        to_em = cue.snapshot.emotion if cue.snapshot else None
        self._shifts.append(
            EmotionShift(
                ts=now,
                stage=str(stage),
                role=str(role),
                text_excerpt=cue.text_excerpt or prev.text_excerpt,
                score=score,
                from_emotion=from_em,
                to_emotion=to_em,
                from_signal=_signal_compact(prev.snapshot),
                to_signal=_signal_compact(cue.snapshot),
                from_summary=prev.summary,
                to_summary=cue.summary,
            )
        )

    def to_prompt(self, *, max_items: int = 6) -> str | None:
        items = list(self._shifts)[-max(0, int(max_items)) :]
        if not items:
            return None

        lines: list[str] = []
        lines.append("最近出现过的“明显情绪变化”（只记录变化较大的几次）：")
        for i, it in enumerate(items, start=1):
            who = "用户" if it.role == "user" else "助手" if it.role == "assistant" else it.role
            stage = it.stage
            lines.append(
                f"{i}) [{who}/{stage}] 信号 {it.from_signal}→{it.to_signal}（变化={it.score:.2f}）"
                f"；触发语句: “{it.text_excerpt}”"
            )
        return "\n".join(lines).strip()


def _signal_compact(snapshot: EmotionSnapshot | None) -> str:
    if snapshot is None or not snapshot.face_detected:
        return "no_face"
    valence = "pos" if snapshot.valence > 0.18 else "neg" if snapshot.valence < -0.18 else "neutral"
    arousal = "high" if snapshot.arousal > 0.60 else "mid" if snapshot.arousal > 0.35 else "low"
    dominant = "unknown" if snapshot.uncertain else snapshot.emotion
    return f"{valence}/{arousal}/{dominant}"
