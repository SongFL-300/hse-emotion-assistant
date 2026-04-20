from __future__ import annotations

from typing import Iterable


EMOTIONS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

_LABEL_ALIASES = {
    "scared": "fear",
    "fearful": "fear",
    "surprised": "surprise",
}

_VALENCE_WEIGHTS = {
    "angry": -0.90,
    "disgust": -0.82,
    "fear": -0.88,
    "happy": 0.95,
    "sad": -0.82,
    "surprise": 0.10,
    "neutral": 0.0,
}

_AROUSAL_WEIGHTS = {
    "angry": 0.90,
    "disgust": 0.55,
    "fear": 0.90,
    "happy": 0.60,
    "sad": 0.35,
    "surprise": 0.80,
    "neutral": 0.10,
}


def normalize_label(label: str | None) -> str:
    raw = (label or "").strip().lower()
    raw = _LABEL_ALIASES.get(raw, raw)
    return raw if raw in EMOTIONS else "unknown"


def normalize_score_dict(scores: dict[str, float] | None) -> dict[str, float]:
    normalized = {name: 0.0 for name in EMOTIONS}
    if not scores:
        return normalized

    total = 0.0
    for key, value in scores.items():
        label = normalize_label(key)
        if label == "unknown":
            continue
        v = max(0.0, float(value))
        normalized[label] += v
        total += v

    if total <= 1e-9:
        return normalized

    return {name: value / total for name, value in normalized.items()}


def compute_valence_arousal(scores: dict[str, float]) -> tuple[float, float]:
    valence = sum(float(scores.get(name, 0.0)) * _VALENCE_WEIGHTS[name] for name in EMOTIONS)
    arousal = sum(float(scores.get(name, 0.0)) * _AROUSAL_WEIGHTS[name] for name in EMOTIONS)
    valence = max(-1.0, min(1.0, float(valence)))
    arousal = max(0.0, min(1.0, float(arousal)))
    return valence, arousal


def classify_affect(
    scores: dict[str, float],
    *,
    neutral_floor: float = 0.26,
    neutral_margin: float = 0.04,
    min_confidence: float = 0.28,
    min_margin: float = 0.05,
) -> tuple[str, float, bool]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered:
        return "unknown", 0.0, True

    top_label, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    neutral_score = float(scores.get("neutral", 0.0))
    margin = float(top_score - second_score)

    if neutral_score >= neutral_floor and (top_label != "neutral" and top_score - neutral_score <= neutral_margin):
        return "neutral", neutral_score, False

    if top_label == "neutral" and top_score >= max(0.30, min_confidence - 0.06):
        return "neutral", float(top_score), False

    if top_score < min_confidence or margin < min_margin:
        if neutral_score >= 0.28:
            return "neutral", neutral_score, False
        return "unknown", float(top_score), True

    return str(top_label), float(top_score), False


def summarise_topk(scores: dict[str, float], *, k: int = 3) -> str:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(k))]
    return ", ".join(f"{name}:{value:.2f}" for name, value in ordered)


def dominant_from_scores(scores: dict[str, float]) -> tuple[str, float]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered:
        return "unknown", 0.0
    return str(ordered[0][0]), float(ordered[0][1])
