from __future__ import annotations

from pathlib import Path
import tempfile
import threading
from typing import Any

from emotion_analyzer.common import classify_affect, compute_valence_arousal, normalize_score_dict


class LibreFaceUnavailableError(RuntimeError):
    pass


class LibreFaceAdapter:
    EXPRESSION_MAP = {
        "neutral": {"neutral": 0.54, "happy": 0.06, "sad": 0.12, "surprise": 0.05, "angry": 0.08, "fear": 0.08, "disgust": 0.07},
        "happiness": {"happy": 0.82, "neutral": 0.12, "surprise": 0.03, "sad": 0.01, "angry": 0.01, "fear": 0.01, "disgust": 0.0},
        "sadness": {"sad": 0.82, "neutral": 0.08, "fear": 0.05, "disgust": 0.02, "angry": 0.02, "surprise": 0.01, "happy": 0.0},
        "surprise": {"surprise": 0.72, "fear": 0.12, "happy": 0.05, "neutral": 0.06, "sad": 0.02, "angry": 0.02, "disgust": 0.01},
        "fear": {"fear": 0.76, "surprise": 0.12, "sad": 0.05, "neutral": 0.03, "angry": 0.02, "disgust": 0.01, "happy": 0.01},
        "disgust": {"disgust": 0.72, "angry": 0.14, "sad": 0.06, "neutral": 0.04, "fear": 0.02, "surprise": 0.01, "happy": 0.01},
        "anger": {"angry": 0.78, "disgust": 0.08, "fear": 0.04, "sad": 0.05, "neutral": 0.03, "surprise": 0.01, "happy": 0.01},
        "contempt": {"disgust": 0.36, "angry": 0.30, "neutral": 0.14, "sad": 0.10, "fear": 0.04, "surprise": 0.02, "happy": 0.04},
    }

    def __init__(self, *, preferred_device: str | None = None):
        self._preferred_device = preferred_device or self._pick_device()
        self.last_fallback_notice = None
        self._weights_dir = Path(".libreface_weights")
        self._temp_dir = Path(tempfile.gettempdir()) / "hsemotion_libreface"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

        try:
            import cv2  # noqa: F401
            from libreface.AU_Recognition.inference import get_au_intensities_and_detect_aus  # type: ignore
            from libreface.Facial_Expression_Recognition.inference import get_facial_expression  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise LibreFaceUnavailableError(f"LibreFace 不可用：{exc}") from exc

        self._get_au_intensities_and_detect_aus = get_au_intensities_and_detect_aus
        self._get_facial_expression = get_facial_expression
        if self._preferred_device == "cpu":
            self.last_fallback_notice = "因为当前环境未获得可用 CUDA，LibreFace 已回退到 CPU 推理。"

    @property
    def preferred_device(self) -> str:
        return self._preferred_device

    def analyze(self, aligned_face_rgb: Any) -> dict[str, Any] | None:
        if aligned_face_rgb is None:
            return None
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise LibreFaceUnavailableError(f"OpenCV 不可用：{exc}") from exc

        image_path = self._temp_dir / f"libreface_{threading.get_ident()}.png"
        with self._write_lock:
            try:
                bgr = cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), bgr)
            except Exception as exc:  # noqa: BLE001
                raise LibreFaceUnavailableError(f"写入 LibreFace 临时图像失败：{exc}") from exc

        try:
            detected_aus, au_intensities = self._get_au_intensities_and_detect_aus(
                str(image_path),
                device=self._preferred_device,
                weights_download_dir=str(self._weights_dir),
            )
            facial_expression = self._get_facial_expression(
                str(image_path),
                device=self._preferred_device,
                weights_download_dir=str(self._weights_dir),
            )
        except Exception as exc:  # noqa: BLE001
            raise LibreFaceUnavailableError(f"LibreFace 推理失败：{exc}") from exc

        expr_key = str(facial_expression or "Neutral").strip().lower()
        intensity_dict = {str(k): float(v) for k, v in (au_intensities or {}).items()}
        presence_dict = {str(k): float(v) for k, v in (detected_aus or {}).items()}
        distribution = normalize_score_dict(self.EXPRESSION_MAP.get(expr_key, self.EXPRESSION_MAP["neutral"]))
        distribution = self._blend_with_au(distribution, intensity_dict=intensity_dict)
        label, prob, uncertain = classify_affect(distribution)
        valence, arousal = compute_valence_arousal(distribution)
        confidence_hint = min(0.92, max(0.28, float(prob) * 0.78 + min(0.26, _mean(intensity_dict.values()) / 6.0)))
        return {
            "emotion": label,
            "probability": float(confidence_hint),
            "all_scores": distribution,
            "input_roi": bgr,
            "valence": float(valence),
            "arousal": float(arousal),
            "uncertain": bool(uncertain),
            "detected_aus": presence_dict,
            "au_intensities": intensity_dict,
            "raw_expression": str(facial_expression or "Neutral"),
        }

    def _blend_with_au(self, base: dict[str, float], *, intensity_dict: dict[str, float]) -> dict[str, float]:
        if not intensity_dict:
            return base
        blended = dict(base)
        norm = {k: max(0.0, min(1.0, float(v) / 5.0)) for k, v in intensity_dict.items()}
        au01 = norm.get("AU01", 0.0)
        au02 = norm.get("AU02", 0.0)
        au04 = norm.get("AU04", 0.0)
        au05 = norm.get("AU05", 0.0)
        au06 = norm.get("AU06", 0.0)
        au07 = norm.get("AU07", 0.0)
        au12 = norm.get("AU12", 0.0)
        au15 = norm.get("AU15", 0.0)
        au17 = norm.get("AU17", 0.0)
        au20 = norm.get("AU20", 0.0)
        au23 = norm.get("AU23", 0.0)
        au24 = norm.get("AU24", 0.0)
        au25 = norm.get("AU25", 0.0)
        au26 = norm.get("AU26", 0.0)

        positive_boost = 0.20 * au12 + 0.16 * au06
        sad_boost = 0.18 * au01 + 0.18 * au15 + 0.12 * au17
        angry_boost = 0.22 * au04 + 0.14 * au23 + 0.10 * au24 + 0.08 * au07
        fear_boost = 0.12 * au01 + 0.12 * au02 + 0.20 * au05 + 0.14 * au20 + 0.10 * au25
        surprise_boost = 0.12 * au01 + 0.12 * au02 + 0.16 * au05 + 0.18 * au26 + 0.08 * au25
        neutral_guard = max(0.0, 0.24 - (positive_boost + sad_boost + angry_boost + fear_boost + surprise_boost))

        negative_total = sad_boost + angry_boost + fear_boost + 0.8 * norm.get("AU09", 0.0) + 0.9 * norm.get("AU10", 0.0)
        positive_total = positive_boost + 0.05 * surprise_boost
        dominance = negative_total - positive_total

        blended["happy"] += positive_boost
        blended["sad"] += sad_boost + 0.10 * au20
        blended["angry"] += angry_boost + 0.08 * au17
        blended["fear"] += fear_boost + 0.06 * au07
        blended["surprise"] += surprise_boost
        blended["disgust"] += 0.12 * norm.get("AU09", 0.0) + 0.14 * norm.get("AU10", 0.0)
        blended["neutral"] += max(0.0, neutral_guard - max(0.0, dominance) * 0.42)
        if dominance > 0.04:
            blended["sad"] += 0.08 * dominance
            blended["angry"] += 0.04 * dominance
            blended["fear"] += 0.04 * dominance
        return normalize_score_dict(blended)

    def recognize_clip(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {}
        valence = _mean(sample.get("valence", 0.0) for sample in samples)
        arousal = _mean(sample.get("arousal", 0.0) for sample in samples)
        confidence = _mean(sample.get("confidence", 0.0) for sample in samples)
        dominant = _mode([str(sample.get("emotion", "unknown")) for sample in samples])
        au_acc: dict[str, list[float]] = {}
        for sample in samples:
            for key, value in (sample.get("au_intensities") or {}).items():
                au_acc.setdefault(str(key), []).append(float(value))
        top_aus = sorted(((k, _mean(v)) for k, v in au_acc.items()), key=lambda item: item[1], reverse=True)[:4]
        return {
            "emotion": dominant,
            "confidence": float(confidence),
            "valence": float(valence),
            "arousal": float(arousal),
            "top_aus": top_aus,
        }

    def draw_result(self, frame: Any, result: dict[str, Any], position: tuple[int, int] = (10, 30)) -> Any:
        try:
            import cv2  # type: ignore
        except Exception:
            return frame
        label = str(result.get("emotion", "unknown"))
        prob = float(result.get("probability", 0.0))
        expr = str(result.get("raw_expression", "")).strip()
        text = f"LibreFace {label}:{prob:.2f}"
        if expr:
            text += f" ({expr})"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(frame, (position[0] - 5, position[1] - text_h - 5), (position[0] + text_w + 5, position[1] + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)
        return frame

    def _pick_device(self) -> str:
        try:
            import torch  # type: ignore
            if bool(torch.cuda.is_available()):
                return "cuda:0"
        except Exception:
            pass
        return "cpu"


def _mean(values) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _mode(values: list[str]) -> str:
    if not values:
        return "unknown"
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
