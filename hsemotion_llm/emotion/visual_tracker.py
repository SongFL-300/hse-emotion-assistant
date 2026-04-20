from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import threading
import time
from typing import Any

from ..config import EmotionConfig
from ..micro_expression.rmes_adapter import RMESAdapter
from .blink_tracker import BlinkTracker
from .libreface_adapter import LibreFaceAdapter, LibreFaceUnavailableError
from .state import EmotionSnapshot, EmotionTimeline, normalize_emotion_label
from .structured import BlinkSignal, StructuredContextSnapshot, WindowSample, WindowStats, compute_rolling_summary, compute_window_stats, format_compact_context_cn, format_rmes_debug_cn, format_rolling_summary_cn, format_window_stats_cn
from emotion_analyzer.common import classify_affect, compute_valence_arousal, normalize_score_dict


class EmotionPipelineUnavailableError(RuntimeError):
    """Raised when the local vision emotion pipeline cannot be initialized."""


@dataclass(frozen=True)
class VisualEmotionStatus:
    timeline: EmotionTimeline
    last_error: str | None
    model_ready: bool
    monitoring: bool
    backend: str
    capture_fps: float = 0.0
    infer_fps: float = 0.0
    preview_fps: float = 0.0


class VisualEmotionTracker:
    def __init__(self, config: EmotionConfig):
        self._config = config
        self._timeline = EmotionTimeline(window_seconds=8.0)

        self._lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._frame_lock = threading.Lock()

        self._stop = threading.Event()
        self._monitoring = threading.Event()
        self._model_ready = threading.Event()

        self._prepare_thread: threading.Thread | None = None
        self._thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None

        self._last_error: str | None = None
        self._cap: Any = None
        self._detector: Any = None
        self._analyzer: Any = None
        self._backend_name = config.engine

        self._latest_frame_bgr: Any = None
        self._latest_raw_frame_bgr: Any = None
        self._capture_frame_bgr: Any = None
        self._latest_pose: dict[str, float] = {}
        self._latest_face_scale = 0.0
        self._latest_face_scale_bucket = "unknown"
        self._subtle_state: dict[str, float] = {}
        self._subtle_cues: list[dict[str, float | str]] = []
        self._subtle_history = deque(maxlen=48)
        self._capture_fps = 0.0
        self._infer_fps = 0.0
        self._preview_fps = 0.0

        self._blink_tracker = BlinkTracker()
        self._rmes = RMESAdapter()

        self._completed_windows: deque[WindowStats] = deque(maxlen=max(4, int(config.window_count)))
        self._active_window_id = 1
        self._active_window_mode = "startup"
        self._active_window_reason = "greeting"
        self._active_window_started_ts = time.time()
        self._active_window_samples: list[WindowSample] = []
        self._active_window_blink_times: list[float] = []
        self._active_window_rmes_events: list[Any] = []
        self._active_window_accepted_rmes_events: list[Any] = []
        self._latest_context_snapshot: StructuredContextSnapshot | None = None
        self._notices: deque[str] = deque(maxlen=32)
        self._seen_notices: set[str] = set()

        if not self._config.enabled:
            self._model_ready.set()

    def prepare_async(self) -> None:
        if not self._config.enabled or self._model_ready.is_set():
            return
        if self._prepare_thread and self._prepare_thread.is_alive():
            return
        with self._lock:
            self._last_error = None
        self._prepare_thread = threading.Thread(target=self._prepare_worker, name="VisualEmotionPrepare", daemon=True)
        self._prepare_thread.start()

    def is_model_ready(self) -> bool:
        return True if not self._config.enabled else self._model_ready.is_set()

    def is_preparing(self) -> bool:
        return bool(self._prepare_thread and self._prepare_thread.is_alive())

    def is_monitoring(self) -> bool:
        return self._monitoring.is_set()

    def get_latest_frame_bgr(self) -> Any | None:
        with self._frame_lock:
            frame = self._latest_frame_bgr
            return None if frame is None else frame.copy()

    def get_latest_raw_frame_bgr(self) -> Any | None:
        with self._frame_lock:
            frame = self._latest_raw_frame_bgr
            return None if frame is None else frame.copy()

    def start(self) -> None:
        if not self._config.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, name="VisualEmotionCapture", daemon=True)
        self._capture_thread.start()
        self._thread = threading.Thread(target=self._run_loop, name="VisualEmotionTracker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._capture_thread = None
        self._thread = None
        self._monitoring.clear()
        self._release_camera()
        with self._frame_lock:
            self._latest_frame_bgr = None
            self._latest_raw_frame_bgr = None
            self._capture_frame_bgr = None

    def shutdown(self) -> None:
        self.stop()
        self._release_models()

    def set_backend(self, backend: str) -> None:
        backend = (backend or "").strip().lower()
        if backend not in {"libreface", "tf_mini_xception", "deepface"}:
            raise ValueError(f"未知视觉后端：{backend}")
        needs_restart = self.is_monitoring()
        self.stop()
        self._release_models()
        self._backend_name = backend
        self._config = replace(self._config, engine=backend)
        self._blink_tracker.reset()
        self._rmes.reset()
        self._subtle_state = {}
        self._subtle_cues = []
        self.prepare_async()
        if needs_restart:
            self.start()

    def get_backend(self) -> str:
        return self._backend_name

    def get_backend_display_name(self) -> str:
        mapping = {
            "libreface": "LibreFace AU/FER 主后端",
            "tf_mini_xception": "Legacy Mini-Xception",
            "deepface": "DeepFace",
        }
        return mapping.get(self._backend_name, self._backend_name)

    def begin_dialogue_window(self, *, mode: str, reason: str) -> WindowStats | None:
        with self._lock:
            finalized = self._finalize_active_window(now=time.time())
            self._active_window_id += 1
            self._active_window_mode = mode
            self._active_window_reason = reason
            self._active_window_started_ts = time.time()
            self._active_window_samples = []
            self._active_window_blink_times = []
            self._active_window_rmes_events = []
            self._active_window_accepted_rmes_events = []
            self._latest_context_snapshot = self._build_context_snapshot_locked()
            return finalized

    def get_summary(self) -> str | None:
        with self._lock:
            summary = self._timeline.summary()
            current = self._latest_context_snapshot.current_window if self._latest_context_snapshot else None
            if not summary:
                return None
            if current is None:
                return summary
            return f"{summary}；窗口#{current.window_id} valence_mean={current.valence_mean:+.2f} arousal_mean={current.arousal_mean:.2f}"

    def get_structured_signal(self) -> str | None:
        with self._lock:
            snapshot = self._build_context_snapshot_locked()
            self._latest_context_snapshot = snapshot
            return format_compact_context_cn(snapshot)

    def get_current_window_text(self) -> str:
        with self._lock:
            snapshot = self._build_context_snapshot_locked()
            self._latest_context_snapshot = snapshot
            return format_window_stats_cn(snapshot.current_window)

    def get_recent_windows_text(self) -> str:
        with self._lock:
            snapshot = self._build_context_snapshot_locked()
            self._latest_context_snapshot = snapshot
            return format_rolling_summary_cn(snapshot.rolling_summary)

    def get_rmes_debug_text(self) -> str:
        with self._lock:
            snapshot = self._build_context_snapshot_locked()
            self._latest_context_snapshot = snapshot
            events = snapshot.accepted_micro_expression_events if snapshot.accepted_for_mainline else snapshot.experimental_micro_expression_events
            return format_rmes_debug_cn(events, accepted=snapshot.accepted_for_mainline)

    def get_subtle_cues(self) -> list[dict[str, float | str]]:
        with self._lock:
            return [dict(item) for item in self._subtle_cues]

    def get_subtle_cues_signal(self) -> str | None:
        with self._lock:
            if not self._subtle_cues:
                return None
            lines = ["[subtle_expression_cues]"]
            for item in self._subtle_cues:
                lines.append(f"- name={item['name']}; direction={item['direction']}; strength={float(item['strength']):.2f}; delta={float(item['delta']):+.3f}")
            lines.append("[/subtle_expression_cues]")
            return "\n".join(lines)

    def get_status(self) -> VisualEmotionStatus:
        with self._lock:
            return VisualEmotionStatus(
                timeline=self._timeline,
                last_error=self._last_error,
                model_ready=self.is_model_ready(),
                monitoring=self.is_monitoring(),
                backend=self._backend_name,
                capture_fps=float(self._capture_fps),
                infer_fps=float(self._infer_fps),
                preview_fps=float(self._preview_fps),
            )

    def get_and_clear_notices(self) -> list[str]:
        with self._lock:
            items = list(self._notices)
            self._notices.clear()
            return items

    def _prepare_worker(self) -> None:
        try:
            self._ensure_models_ready()
        except Exception:
            return

    def _ensure_models_ready(self) -> None:
        if not self._config.enabled:
            self._model_ready.set()
            return
        if self._model_ready.is_set():
            return
        with self._init_lock:
            if self._model_ready.is_set():
                return
            try:
                self._prepare_models()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                raise
            else:
                with self._lock:
                    self._last_error = None
                self._model_ready.set()

    def _prepare_models(self) -> None:
        try:
            import cv2  # noqa: F401
        except Exception as exc:
            raise EmotionPipelineUnavailableError("OpenCV not available; please install opencv-python.") from exc
        try:
            from face_mesh_detector.FaceMeshDetector import FaceMeshDetector  # type: ignore
        except Exception as exc:
            raise EmotionPipelineUnavailableError("Cannot import FaceMeshDetector.") from exc

        if self._backend_name == "deepface":
            from emotion_analyzer.DeepFaceAnalyzer import DeepFaceAnalyzer  # type: ignore
            analyzer = DeepFaceAnalyzer()
        elif self._backend_name == "libreface":
            try:
                analyzer = LibreFaceAdapter()
            except LibreFaceUnavailableError as exc:
                raise EmotionPipelineUnavailableError(str(exc)) from exc
        else:
            from emotion_analyzer.EmotionAnalyzer import EmotionAnalyzer  # type: ignore
            analyzer = EmotionAnalyzer()

        detector = FaceMeshDetector(max_num_faces=1, output_size=(112, 112), offline_mode=self._config.mediapipe_offline_mode)
        self._detector = detector
        self._analyzer = analyzer
        notice = getattr(analyzer, "last_fallback_notice", None)
        if notice:
            self._push_notice(str(notice))

    def _ensure_camera(self) -> Any:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise EmotionPipelineUnavailableError("OpenCV not available; please install opencv-python.") from exc

        with self._init_lock:
            if self._cap is not None:
                try:
                    if self._cap.isOpened():
                        return self._cap
                except Exception:
                    pass
            cap = cv2.VideoCapture(self._config.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(self._config.camera_index)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                raise EmotionPipelineUnavailableError(f"Cannot open camera index={self._config.camera_index}.")
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, max(60.0, float(self._config.sample_fps)))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self._cap = cap
            return cap

    def _capture_loop(self) -> None:
        try:
            cap = self._ensure_camera()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return

        last_stat_ts = time.time()
        capture_count = 0
        try:
            import cv2  # type: ignore
        except Exception:
            return

        while not self._stop.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                capture_count += 1
                frame = cv2.flip(frame, 1)
                with self._frame_lock:
                    self._capture_frame_bgr = frame.copy()
                    self._latest_raw_frame_bgr = frame.copy()
                    # Let UI consume fast raw frames even when inference is slower.
                    self._latest_frame_bgr = frame.copy()
                now = time.time()
                if now - last_stat_ts >= 1.0:
                    elapsed = max(1e-6, now - last_stat_ts)
                    with self._lock:
                        self._capture_fps = float(capture_count / elapsed)
                    capture_count = 0
                    last_stat_ts = now
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                time.sleep(0.01)

    def _release_camera(self) -> None:
        cap = self._cap
        self._cap = None
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

    def _release_models(self) -> None:
        detector = self._detector
        self._detector = None
        self._analyzer = None
        self._model_ready.clear()
        try:
            if detector is not None:
                detector.release()
        except Exception:
            pass

    def _run_loop(self) -> None:
        try:
            self._ensure_models_ready()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return

        try:
            import cv2  # type: ignore
        except Exception:
            with self._lock:
                self._last_error = "OpenCV not available."
            return

        infer_fps = max(60.0, float(self._config.sample_fps))
        infer_interval = 1.0 / max(1e-6, infer_fps)
        preview_interval = 1.0 / max(60.0, infer_fps)
        last_preview_ts = 0.0
        last_infer_ts = 0.0
        last_stat_ts = time.time()
        infer_count = 0
        preview_count = 0
        last_landmarks = None
        last_aligned = None
        last_result = None
        last_face_detected = False

        self._monitoring.set()
        try:
            while not self._stop.is_set():
                now = time.time()
                if now - last_preview_ts < preview_interval:
                    time.sleep(0.002)
                    continue
                last_preview_ts = now
                try:
                    with self._frame_lock:
                        frame = None if self._capture_frame_bgr is None else self._capture_frame_bgr.copy()
                    if frame is None:
                        time.sleep(0.005)
                        continue
                    raw_frame = frame.copy()

                    if now - last_infer_ts >= infer_interval:
                        last_infer_ts = now
                        infer_count += 1
                        aligned_face, landmarks = self._detector.align_face(frame)
                        if aligned_face is None:
                            last_face_detected = False
                            last_landmarks = None
                            last_aligned = None
                            last_result = None
                            snap = EmotionSnapshot(ts=now, emotion="unknown", probability=0.0, scores=None, face_detected=False)
                            with self._lock:
                                self._timeline.add(snap)
                        else:
                            last_face_detected = True
                            last_landmarks = landmarks
                            last_aligned = aligned_face
                            pose = self._detector.estimate_pose(landmarks, frame.shape)
                            face_scale = self._estimate_face_scale(landmarks, frame.shape)
                            face_scale_bucket = self._bucket_face_scale(face_scale)
                            self._latest_pose = {str(k): float(v) for k, v in pose.items()}
                            self._latest_face_scale = float(face_scale)
                            self._latest_face_scale_bucket = face_scale_bucket
                            blink_obs = self._blink_tracker.observe(landmarks, now, frontal_score=float(pose.get("frontal_score", 0.0)))
                            subtle_metrics = self._extract_subtle_metrics(landmarks)
                            subtle_cues = self._update_subtle_cues(subtle_metrics, frontal_score=float(pose.get("frontal_score", 1.0)))
                            result = self._analyzer.analyze(aligned_face) if self._analyzer is not None else None
                            notice = getattr(self._analyzer, "last_fallback_notice", None) if self._analyzer is not None else None
                            if notice:
                                self._push_notice(str(notice))
                            last_result = result
                            if not result:
                                snap = EmotionSnapshot(ts=now, emotion="unknown", probability=0.0, scores=None, face_detected=True, uncertain=True, pose_yaw=float(pose.get("yaw", 0.0)), pose_pitch=float(pose.get("pitch", 0.0)), pose_roll=float(pose.get("roll", 0.0)), frontal_score=float(pose.get("frontal_score", 0.0)))
                                au_intensities: dict[str, float] = {}
                            else:
                                label = normalize_emotion_label(str(result.get("emotion", "neutral")))
                                prob = float(result.get("probability", 0.0))
                                raw_scores = result.get("all_scores") or {}
                                scores = {normalize_emotion_label(k): float(v) for k, v in raw_scores.items()}
                                scores = self._refine_scores_with_subtle_metrics(
                                    scores=scores,
                                    subtle_metrics=subtle_metrics,
                                    frontal_score=float(pose.get("frontal_score", 1.0)),
                                    face_scale=float(face_scale),
                                )
                                label, prob, uncertain = classify_affect(scores)
                                valence, arousal = compute_valence_arousal(scores)
                                frontal_score = float(pose.get("frontal_score", 1.0))
                                uncertain = bool(result.get("uncertain", False)) or bool(uncertain)
                                if frontal_score < 0.42:
                                    prob *= max(0.38, frontal_score)
                                    uncertain = True
                                    if label not in {"neutral", "unknown"}:
                                        label = "neutral" if frontal_score >= 0.30 else "unknown"
                                elif frontal_score < 0.60:
                                    prob *= 0.72 + 0.28 * frontal_score
                                    uncertain = uncertain or prob < 0.42
                                elif frontal_score < 0.78:
                                    prob *= 0.90 + 0.10 * frontal_score
                                snap = EmotionSnapshot(ts=now, emotion=label, probability=prob, scores=scores, face_detected=True, valence=float(valence), arousal=float(arousal), uncertain=uncertain, pose_yaw=float(pose.get("yaw", 0.0)), pose_pitch=float(pose.get("pitch", 0.0)), pose_roll=float(pose.get("roll", 0.0)), frontal_score=frontal_score)
                                au_intensities = {str(k): float(v) for k, v in (result.get("au_intensities") or {}).items()}
                            with self._lock:
                                self._timeline.add(snap)
                                self._subtle_cues = subtle_cues
                                if subtle_cues:
                                    self._subtle_history.append((now, [dict(x) for x in subtle_cues]))
                                self._record_window_sample(snap=snap, blink_event=blink_obs.blink_event, subtle_cues=subtle_cues, au_intensities=au_intensities)
                                if blink_obs.blink_event:
                                    self._active_window_blink_times.append(now)
                                if self._config.rmes_enabled:
                                    event = self._rmes.observe(ts=now, window_id=self._active_window_id, subtle_metrics=subtle_metrics, pose=pose, dominant_emotion=snap.emotion, valence=snap.valence, arousal=snap.arousal, confidence=snap.probability, au_intensities=au_intensities)
                                    if event is not None:
                                        self._active_window_rmes_events.append(event)
                                        if not self._config.rmes_acceptance_gate and self._config.rmes_mode == "production_candidate":
                                            self._active_window_accepted_rmes_events.append(event)
                                        elif self._config.rmes_mode == "production_candidate" and self._rmes.is_accepted_for_mainline():
                                            self._active_window_accepted_rmes_events.append(event)
                                self._last_error = None
                                self._latest_context_snapshot = self._build_context_snapshot_locked()

                    annotated = self._annotate_for_display(frame, aligned_face_rgb=last_aligned, landmarks_px=last_landmarks, result=last_result, face_detected=last_face_detected)
                    preview_count += 1
                    with self._frame_lock:
                        self._latest_frame_bgr = annotated
                    if now - last_stat_ts >= 1.0:
                        elapsed = max(1e-6, now - last_stat_ts)
                        with self._lock:
                            self._infer_fps = float(infer_count / elapsed)
                            self._preview_fps = float(preview_count / elapsed)
                        last_stat_ts = now
                        infer_count = 0
                        preview_count = 0
                except Exception as exc:
                    with self._lock:
                        self._last_error = str(exc)
                    time.sleep(0.05)
        finally:
            self._monitoring.clear()

    def _record_window_sample(
        self,
        *,
        snap: EmotionSnapshot,
        blink_event: bool,
        subtle_cues: list[dict[str, float | str]],
        au_intensities: dict[str, float],
    ) -> None:
        self._active_window_samples.append(
            WindowSample(
                ts=snap.ts,
                valence=snap.valence,
                arousal=snap.arousal,
                confidence=snap.probability,
                frontal_score=snap.frontal_score,
                face_scale=float(self._latest_face_scale),
                face_scale_bucket=str(self._latest_face_scale_bucket),
                dominant_emotion=snap.emotion,
                au_intensity=au_intensities,
                subtle_cues=[dict(item) for item in subtle_cues],
                blink_event=blink_event,
                uncertain=snap.uncertain,
            )
        )

    def _finalize_active_window(self, *, now: float) -> WindowStats | None:
        if not self._active_window_samples and not self._active_window_rmes_events and not self._active_window_accepted_rmes_events:
            return None
        previous = self._completed_windows[-1] if self._completed_windows else None
        stats = compute_window_stats(window_id=self._active_window_id, mode=self._active_window_mode, reason=self._active_window_reason, start_ts=self._active_window_started_ts, end_ts=now, samples=self._active_window_samples, blink_times=self._active_window_blink_times, experimental_events=self._active_window_rmes_events, accepted_events=self._active_window_accepted_rmes_events)
        blink_signal = stats.blink_signal
        prev_blink = previous.blink_signal.blink_rate_per_min if previous else 0.0
        baseline = [w.blink_signal.blink_rate_per_min for w in self._completed_windows]
        baseline_mean = sum(baseline) / len(baseline) if baseline else 0.0
        blink_signal = BlinkSignal(blink_count=blink_signal.blink_count, blink_rate_per_min=blink_signal.blink_rate_per_min, mean_inter_blink_interval=blink_signal.mean_inter_blink_interval, blink_rate_delta_vs_prev_window=float(blink_signal.blink_rate_per_min - prev_blink), blink_rate_delta_vs_rolling_baseline=float(blink_signal.blink_rate_per_min - baseline_mean), blink_variability=blink_signal.blink_variability)
        stats = replace(stats, blink_signal=blink_signal)
        self._completed_windows.append(stats)
        return stats

    def _build_context_snapshot_locked(self) -> StructuredContextSnapshot:
        current = compute_window_stats(window_id=self._active_window_id, mode=self._active_window_mode, reason=self._active_window_reason, start_ts=self._active_window_started_ts, end_ts=time.time(), samples=self._active_window_samples, blink_times=self._active_window_blink_times, experimental_events=self._active_window_rmes_events, accepted_events=self._active_window_accepted_rmes_events)
        rolling = compute_rolling_summary(list(self._completed_windows) + ([current] if current.sample_count else []), max_windows=max(1, int(self._config.window_count)))
        accepted_for_mainline = self._config.rmes_mode == "production_candidate" and ((not self._config.rmes_acceptance_gate) or self._rmes.is_accepted_for_mainline())
        accepted_events = self._active_window_accepted_rmes_events if accepted_for_mainline else []
        pose_quality = dict(self._latest_pose)
        pose_quality["face_scale"] = float(self._latest_face_scale)
        return StructuredContextSnapshot(
            backend=self._backend_name,
            backend_description=self.get_backend_display_name(),
            current_window=current if current.sample_count else None,
            rolling_summary=rolling,
            accepted_micro_expression_events=list(accepted_events),
            experimental_micro_expression_events=list(self._active_window_rmes_events),
            subtle_cues=[dict(item) for item in self._subtle_cues],
            pose_quality=pose_quality,
            face_scale=float(self._latest_face_scale),
            face_scale_bucket=str(self._latest_face_scale_bucket),
            accepted_for_mainline=accepted_for_mainline,
        )

    def _build_input_roi_display(self, result: Any) -> Any | None:
        if not result or not isinstance(result, dict):
            return None
        roi = result.get("input_roi")
        if roi is None:
            return None
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return None
        if not isinstance(roi, np.ndarray):
            return None
        arr = roi
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return cv2.resize(arr, (112, 112), interpolation=cv2.INTER_AREA)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.resize(arr_bgr, (112, 112), interpolation=cv2.INTER_AREA)

    def _annotate_for_display(self, frame_bgr: Any, *, aligned_face_rgb: Any | None, landmarks_px: Any | None, result: Any | None, face_detected: bool) -> Any:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return frame_bgr
        out = frame_bgr.copy()
        if result is not None and hasattr(self._analyzer, "draw_result"):
            try:
                self._analyzer.draw_result(out, result, position=(20, 40))
            except Exception:
                pass
        else:
            text = "No face" if not face_detected else "Analyzing..."
            cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if result and isinstance(result, dict) and result.get("au_intensities"):
            au_text = ", ".join([f"{k}:{v:.2f}" for k, v in sorted(result["au_intensities"].items(), key=lambda kv: kv[1], reverse=True)[:3]])
            if au_text:
                cv2.putText(out, f"AU {au_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1)
        if aligned_face_rgb is not None and float(self._config.sample_fps) < 45.0:
            try:
                aligned_bgr = cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2BGR)
                aligned_show = cv2.resize(aligned_bgr, (112, 112), interpolation=cv2.INTER_AREA)
            except Exception:
                aligned_show = None
            input_roi_display = self._build_input_roi_display(result)
            if input_roi_display is None:
                input_roi_display = np.zeros((112, 112, 3), dtype=np.uint8)
            h, w = out.shape[:2]
            pad = 10
            if h >= 112 + pad and w >= 224 + pad * 2:
                try:
                    out[h - 112 - pad : h - pad, w - 112 - pad : w - pad] = input_roi_display
                    cv2.putText(out, "Model Input", (w - 112 - pad, h - 112 - pad - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    if aligned_show is not None:
                        out[h - 112 - pad : h - pad, w - 224 - pad * 2 : w - 112 - pad * 2] = aligned_show
                        cv2.putText(out, "Aligned RGB", (w - 224 - pad * 2, h - 112 - pad - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                except Exception:
                    pass
        max_w = 720
        if out.shape[1] > max_w:
            scale = max_w / float(out.shape[1])
            new_h = max(1, int(out.shape[0] * scale))
            out = cv2.resize(out, (max_w, new_h), interpolation=cv2.INTER_AREA)
        return out

    def _extract_subtle_metrics(self, landmarks_px: list[tuple[int, int]]) -> dict[str, float]:
        import numpy as np  # type: ignore

        def center(indices):
            pts = np.array([landmarks_px[i] for i in indices], dtype=np.float32)
            return np.mean(pts, axis=0)

        left_eye = center(self._detector.ALIGNMENT_POINTS["left_eye_center"])
        right_eye = center(self._detector.ALIGNMENT_POINTS["right_eye_center"])
        mouth_left = center(self._detector.ALIGNMENT_POINTS["mouth_left"])
        mouth_right = center(self._detector.ALIGNMENT_POINTS["mouth_right"])
        upper_lip = center([13, 312, 311])
        lower_lip = center([14, 87, 178])
        left_brow = center([70, 63, 105, 66, 107])
        right_brow = center([336, 296, 334, 293, 300])
        left_eye_upper = center([159, 160, 161])
        left_eye_lower = center([145, 144, 153])
        right_eye_upper = center([386, 387, 388])
        right_eye_lower = center([374, 380, 381])

        eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
        mouth_width = max(1.0, float(np.linalg.norm(mouth_right - mouth_left)))
        return {
            "eye_open": float((np.linalg.norm(left_eye_upper - left_eye_lower) + np.linalg.norm(right_eye_upper - right_eye_lower)) / (2.0 * eye_dist)),
            "eye_asymmetry": float(abs(np.linalg.norm(left_eye_upper - left_eye_lower) - np.linalg.norm(right_eye_upper - right_eye_lower)) / eye_dist),
            "mouth_open": float(np.linalg.norm(upper_lip - lower_lip) / mouth_width),
            "brow_raise": float(((left_eye[1] - left_brow[1]) + (right_eye[1] - right_brow[1])) / (2.0 * eye_dist)),
            "brow_asymmetry": float(abs((left_eye[1] - left_brow[1]) - (right_eye[1] - right_brow[1])) / eye_dist),
            "mouth_corner_tilt": float(abs(mouth_left[1] - mouth_right[1]) / mouth_width),
            "lip_press": float(max(0.0, 0.22 - (np.linalg.norm(upper_lip - lower_lip) / eye_dist))),
        }

    def _refine_scores_with_subtle_metrics(
        self,
        *,
        scores: dict[str, float],
        subtle_metrics: dict[str, float],
        frontal_score: float,
        face_scale: float,
    ) -> dict[str, float]:
        if not scores:
            return scores

        refined = dict(scores)
        if frontal_score < 0.50:
            return normalize_score_dict(refined)

        eye_open = float(subtle_metrics.get("eye_open", 0.0))
        mouth_open = float(subtle_metrics.get("mouth_open", 0.0))
        brow_raise = float(subtle_metrics.get("brow_raise", 0.0))
        lip_press = float(subtle_metrics.get("lip_press", 0.0))
        mouth_corner_tilt = float(subtle_metrics.get("mouth_corner_tilt", 0.0))
        scale_bucket = self._bucket_face_scale(face_scale)

        if scale_bucket == "near":
            mouth_open_target = 0.22
            eye_open_target = 0.11
            brow_raise_target = 0.12
            negative_gain = 1.28
            neutral_cut = 0.28
        elif scale_bucket == "far":
            mouth_open_target = 0.17
            eye_open_target = 0.09
            brow_raise_target = 0.10
            negative_gain = 0.92
            neutral_cut = 0.18
        else:
            mouth_open_target = 0.18
            eye_open_target = 0.095
            brow_raise_target = 0.11
            negative_gain = 1.0
            neutral_cut = 0.22

        sad_signal = max(0.0, (mouth_open_target - mouth_open) * 2.6) + max(0.0, lip_press - 0.01) * 2.2
        sad_signal += max(0.0, brow_raise_target - brow_raise) * 2.0
        low_energy_signal = max(0.0, eye_open_target - eye_open) * 2.2
        negative_signal = min(0.70, (sad_signal + low_energy_signal + max(0.0, mouth_corner_tilt - 0.010) * 1.2) * negative_gain)

        positive_signal = max(0.0, mouth_open - 0.22) * 1.8 + max(0.0, brow_raise - 0.18) * 0.8

        refined["sad"] = float(refined.get("sad", 0.0)) + negative_signal * 0.32
        refined["fear"] = float(refined.get("fear", 0.0)) + max(0.0, 0.12 - eye_open) * 0.06
        refined["neutral"] = max(0.0, float(refined.get("neutral", 0.0)) - negative_signal * neutral_cut)
        refined["happy"] = max(0.0, float(refined.get("happy", 0.0)) - negative_signal * 0.12)

        if positive_signal > 0.08:
            refined["happy"] = float(refined.get("happy", 0.0)) + positive_signal * 0.10

        return normalize_score_dict(refined)

    def _estimate_face_scale(self, landmarks_px: list[tuple[int, int]], frame_shape: tuple[int, ...]) -> float:
        if not landmarks_px:
            return 0.0
        xs = [pt[0] for pt in landmarks_px]
        ys = [pt[1] for pt in landmarks_px]
        width = max(1.0, float(max(xs) - min(xs)))
        height = max(1.0, float(max(ys) - min(ys)))
        frame_h = max(1.0, float(frame_shape[0]))
        frame_w = max(1.0, float(frame_shape[1]))
        return max(0.0, min(1.0, (width * height) / (frame_w * frame_h)))

    def _bucket_face_scale(self, face_scale: float) -> str:
        if face_scale >= 0.22:
            return "near"
        if face_scale <= 0.10:
            return "far"
        return "mid"

    def _update_subtle_cues(self, metrics: dict[str, float], *, frontal_score: float) -> list[dict[str, float | str]]:
        cues: list[dict[str, float | str]] = []
        if frontal_score < 0.45:
            self._subtle_state = metrics
            return cues
        thresholds = {"eye_open": 0.012, "eye_asymmetry": 0.010, "mouth_open": 0.012, "brow_raise": 0.010, "brow_asymmetry": 0.008, "mouth_corner_tilt": 0.008, "lip_press": 0.008}
        name_map = {
            "eye_open": ("eye_widen", "eye_narrow"),
            "eye_asymmetry": ("eye_asymmetry_shift", "eye_asymmetry_shift"),
            "mouth_open": ("mouth_opening", "lip_closing"),
            "brow_raise": ("brow_raise", "brow_lower"),
            "brow_asymmetry": ("brow_asymmetry_shift", "brow_asymmetry_shift"),
            "mouth_corner_tilt": ("mouth_soften", "mouth_flatten"),
            "lip_press": ("lip_press", "lip_release"),
        }
        for key, value in metrics.items():
            prev = float(self._subtle_state.get(key, value))
            delta = float(value - prev)
            self._subtle_state[key] = 0.82 * prev + 0.18 * float(value)
            threshold = float(thresholds.get(key, 0.01))
            if abs(delta) < threshold:
                continue
            up_name, down_name = name_map.get(key, (key, key))
            direction = "up" if delta > 0 else "down"
            cues.append({"name": up_name if delta > 0 else down_name, "direction": direction, "strength": min(0.99, abs(delta) / max(1e-6, threshold * 3.0)), "delta": delta})
        cues.sort(key=lambda item: float(item["strength"]), reverse=True)
        return cues[:5]

    def _push_notice(self, text: str) -> None:
        msg = (text or "").strip()
        if not msg:
            return
        if msg in self._seen_notices:
            return
        self._seen_notices.add(msg)
        self._notices.append(msg)
