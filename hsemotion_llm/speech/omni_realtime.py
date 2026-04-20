from __future__ import annotations

import base64
import audioop
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..config import DashScopeConfig, OmniRealtimeConfig


class OmniRealtimeUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OmniUserTranscriptEvent:
    text: str
    is_final: bool = False


@dataclass(frozen=True)
class OmniAssistantTranscriptEvent:
    text: str
    is_final: bool = False


class OmniRealtimeSession:
    def __init__(
        self,
        *,
        dashscope: DashScopeConfig,
        omni: OmniRealtimeConfig,
        get_frame_bgr: Callable[[], Any | None] | None = None,
        get_visual_signal: Callable[[], str | None] | None = None,
        on_user_transcript: Callable[[OmniUserTranscriptEvent], None] | None = None,
        on_assistant_transcript: Callable[[OmniAssistantTranscriptEvent], None] | None = None,
        on_state: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        self._dashscope_cfg = dashscope
        self._omni_cfg = omni
        self._get_frame_bgr = get_frame_bgr
        self._get_visual_signal = get_visual_signal
        self._on_user_transcript = on_user_transcript
        self._on_assistant_transcript = on_assistant_transcript
        self._on_state = on_state
        self._on_error = on_error

        self._lock = threading.Lock()
        self._running = threading.Event()
        self._video_stop = threading.Event()
        self._fatal_stop = threading.Event()
        self._last_error_text = ""
        self._last_error_ts = 0.0

        self._conversation: Any = None
        self._input_stream: Any = None
        self._output_stream: Any = None
        self._video_thread: threading.Thread | None = None

        self._user_text = ""
        self._assistant_text = ""
        self._audio_started = threading.Event()
        self._assistant_speaking = threading.Event()
        self._last_cancel_ts = 0.0
        self._last_signal_update_ts = 0.0
        self._last_visual_signal = ""
        self._external_knowledge = ""
        self._last_external_knowledge = ""

    def start(self) -> None:
        with self._lock:
            if self._running.is_set():
                return
            self._running.set()
            self._fatal_stop.clear()
            self._last_error_text = ""
            self._last_error_ts = 0.0
            self._last_cancel_ts = 0.0
            self._audio_started.clear()
            self._assistant_speaking.clear()
            self._last_signal_update_ts = 0.0
            self._last_visual_signal = ""
            self._external_knowledge = ""
            self._last_external_knowledge = ""

        try:
            import sounddevice as sd  # type: ignore
            from dashscope.audio.qwen_omni import (  # type: ignore
                AudioFormat,
                MultiModality,
                OmniRealtimeCallback,
                OmniRealtimeConversation,
            )
        except Exception as exc:  # noqa: BLE001
            self._running.clear()
            raise OmniRealtimeUnavailableError(
                "Omni Realtime 依赖不可用，请确认 dashscope 与 sounddevice 已安装。"
            ) from exc

        if self._omni_cfg.input_sample_rate != 16000 or self._omni_cfg.output_sample_rate != 24000:
            self._running.clear()
            raise OmniRealtimeUnavailableError("当前实现仅支持 16k 输入和 24k 输出。")

        session = self

        class _Cb(OmniRealtimeCallback):  # type: ignore[misc]
            def on_open(self) -> None:
                session._emit_state("已连接 Omni Realtime")

            def on_close(self, close_status_code, close_msg) -> None:  # noqa: ANN001
                session._handle_remote_close(close_status_code, close_msg)

            def on_event(self, message: dict) -> None:
                session._handle_event(message)

        try:
            output_stream = sd.RawOutputStream(
                samplerate=self._omni_cfg.output_sample_rate,
                channels=1,
                dtype="int16",
            )
            output_stream.start()

            conv = OmniRealtimeConversation(
                model=self._omni_cfg.model,
                callback=_Cb(),
                url=self._omni_cfg.base_ws_url,
                api_key=self._dashscope_cfg.api_key,
            )
            conv.connect()
            conv.update_session(
                output_modalities=[MultiModality.TEXT, MultiModality.AUDIO],
                voice=self._omni_cfg.voice,
                input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                enable_input_audio_transcription=True,
                input_audio_transcription_model=self._omni_cfg.transcription_model,
                enable_turn_detection=True,
                turn_detection_type="server_vad",
                turn_detection_threshold=0.2,
                turn_detection_silence_duration_ms=800,
                prefix_padding_ms=300,
                instructions=self._build_runtime_instructions(),
            )

            def _audio_cb(indata, frames, time_info, status):  # noqa: ANN001
                if not session._running.is_set():
                    return
                try:
                    if (
                        session._omni_cfg.local_barge_in_enabled
                        and session._assistant_speaking.is_set()
                        and session._should_cancel_for_barge_in(bytes(indata))
                    ):
                        try:
                            conv.cancel_response()
                            session._assistant_speaking.clear()
                            session._emit_state("检测到你重新开口，已尝试打断当前回复…")
                        except Exception as exc:  # noqa: BLE001
                            session._emit_error(f"尝试打断当前回复失败: {exc}")
                    conv.append_audio(base64.b64encode(bytes(indata)).decode("ascii"))
                    session._audio_started.set()
                    session._maybe_refresh_visual_signal()
                except Exception as exc:  # noqa: BLE001
                    session._handle_fatal_error(f"发送麦克风音频失败: {exc}")

            input_stream = sd.RawInputStream(
                samplerate=self._omni_cfg.input_sample_rate,
                blocksize=int(self._omni_cfg.input_sample_rate / 10),
                channels=1,
                dtype="int16",
                callback=_audio_cb,
            )
            input_stream.start()
        except Exception:
            self._running.clear()
            raise

        with self._lock:
            self._conversation = conv
            self._input_stream = input_stream
            self._output_stream = output_stream

        self._emit_state("纯语音全模态模式已启动")

        if self._omni_cfg.enable_video and self._get_frame_bgr:
            self._video_stop.clear()
            self._video_thread = threading.Thread(target=self._video_loop, name="OmniRealtimeVideo", daemon=True)
            self._video_thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running.is_set():
                return
            self._running.clear()
            conv = self._conversation
            input_stream = self._input_stream
            output_stream = self._output_stream
            self._conversation = None
            self._input_stream = None
            self._output_stream = None

        self._video_stop.set()
        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=1.5)
        self._video_thread = None

        try:
            if input_stream is not None:
                input_stream.stop()
                input_stream.close()
        except Exception:
            pass
        try:
            if output_stream is not None:
                output_stream.stop()
                output_stream.close()
        except Exception:
            pass
        try:
            if conv is not None:
                conv.end_session_async()
                conv.close()
        except Exception:
            pass

    def is_running(self) -> bool:
        return self._running.is_set()

    def _video_loop(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._emit_error(f"视频帧发送不可用: {exc}")
            return

        interval = 1.0 / max(0.2, float(self._omni_cfg.video_fps))
        while self._running.is_set() and not self._video_stop.is_set():
            t0 = time.time()
            frame = self._get_frame_bgr() if self._get_frame_bgr else None
            conv = self._conversation
            if frame is not None and conv is not None and self._audio_started.is_set():
                try:
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        conv.append_video(base64.b64encode(encoded.tobytes()).decode("ascii"))
                        self._maybe_refresh_visual_signal()
                except Exception as exc:  # noqa: BLE001
                    self._handle_fatal_error(f"发送视频帧失败: {exc}")
            dt = time.time() - t0
            time.sleep(max(0.0, interval - dt))

    def _handle_event(self, message: dict) -> None:
        event_type = str(message.get("type", ""))

        if event_type == "response.audio.delta":
            delta = message.get("delta")
            if isinstance(delta, str) and delta:
                try:
                    self._assistant_speaking.set()
                    audio = base64.b64decode(delta)
                    if self._output_stream is not None:
                        self._output_stream.write(audio)
                except Exception as exc:  # noqa: BLE001
                    self._emit_error(f"播放模型音频失败: {exc}")
            return

        if event_type in {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"}:
            self._emit_state("正在聆听…" if "started" in event_type else "检测到停顿，等待回复…")
            return

        if event_type in {"input_audio_transcription.delta", "conversation.item.input_audio_transcription.delta"}:
            text = self._extract_text(message)
            if text:
                self._user_text += text
                if self._on_user_transcript:
                    self._on_user_transcript(OmniUserTranscriptEvent(text=self._user_text, is_final=False))
            return

        if event_type in {
            "input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.completed",
        }:
            text = self._extract_text(message)
            if text:
                self._user_text = text
            if self._on_user_transcript and self._user_text:
                self._on_user_transcript(OmniUserTranscriptEvent(text=self._user_text, is_final=True))
            self._assistant_text = ""
            return

        if event_type in {"response.audio_transcript.delta", "response.text.delta"}:
            text = self._extract_text(message)
            if text:
                self._assistant_text += text
                if self._on_assistant_transcript:
                    self._on_assistant_transcript(
                        OmniAssistantTranscriptEvent(text=self._assistant_text, is_final=False)
                    )
            return

        if event_type in {"response.audio_transcript.done", "response.text.done", "response.done"}:
            text = self._extract_text(message) or self._assistant_text
            if text:
                self._assistant_text = text
            if self._on_assistant_transcript and self._assistant_text:
                self._on_assistant_transcript(
                    OmniAssistantTranscriptEvent(text=self._assistant_text, is_final=True)
                )
            self._assistant_speaking.clear()
            self._emit_state("聆听中…（纯语音全模态）")
            self._user_text = ""
            self._assistant_text = ""
            return

        if event_type == "error":
            self._handle_fatal_error(self._extract_error(message))

    def _extract_text(self, message: dict) -> str:
        for key in ("text", "transcript"):
            value = message.get(key)
            if isinstance(value, str) and value:
                return value
        delta = message.get("delta")
        if isinstance(delta, str) and delta and not self._looks_like_base64(delta):
            return delta
        for value in message.values():
            if isinstance(value, dict):
                nested = self._extract_text(value)
                if nested:
                    return nested
        return ""

    def _extract_error(self, message: dict) -> str:
        err = message.get("error")
        if isinstance(err, dict):
            text = err.get("message") or err.get("code")
            if isinstance(text, str):
                return text
        return str(message)

    def _looks_like_base64(self, value: str) -> bool:
        if len(value) < 24:
            return False
        alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
        return all(ch in alphabet for ch in value[:48])

    def _emit_state(self, text: str) -> None:
        if self._on_state:
            self._on_state(text)

    def _emit_error(self, text: str) -> None:
        now = time.time()
        if text == self._last_error_text and now - self._last_error_ts < 3.0:
            return
        self._last_error_text = text
        self._last_error_ts = now
        if self._on_error:
            self._on_error(text)

    def _handle_remote_close(self, close_status_code: Any, close_msg: Any) -> None:
        msg = f"会话已关闭 ({close_status_code})"
        if close_msg:
            msg = f"{msg}: {close_msg}"
        self._shutdown_local(close_conversation=False)
        self._emit_state(msg)

    def _handle_fatal_error(self, text: str) -> None:
        if self._fatal_stop.is_set():
            return
        self._fatal_stop.set()
        self._emit_error(text)
        self._shutdown_local(close_conversation=True)
        self._emit_state("Omni 会话已停止")

    def _shutdown_local(self, *, close_conversation: bool) -> None:
        with self._lock:
            conv = self._conversation
            input_stream = self._input_stream
            output_stream = self._output_stream
            self._conversation = None
            self._input_stream = None
            self._output_stream = None
            self._running.clear()
            self._assistant_speaking.clear()
            self._audio_started.clear()

        self._video_stop.set()
        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=1.5)
        self._video_thread = None

        try:
            if input_stream is not None:
                input_stream.stop()
                input_stream.close()
        except Exception:
            pass
        try:
            if output_stream is not None:
                output_stream.stop()
                output_stream.close()
        except Exception:
            pass
        if close_conversation:
            try:
                if conv is not None:
                    conv.end_session_async()
                    conv.close()
            except Exception:
                pass

    def _should_cancel_for_barge_in(self, audio_bytes: bytes) -> bool:
        now = time.time()
        if now - self._last_cancel_ts < max(0.1, float(self._omni_cfg.local_barge_in_cooldown_s)):
            return False
        try:
            rms = audioop.rms(audio_bytes, 2)
        except Exception:
            return False
        if rms < float(self._omni_cfg.local_barge_in_rms):
            return False
        self._last_cancel_ts = now
        return True

    def _build_runtime_instructions(self) -> str:
        knowledge_block = ""
        if self._external_knowledge:
            knowledge_block = (
                "\n\n下面是本轮用户话题命中的专业知识库片段，仅在相关时自然吸收，不要逐字照搬：\n"
                f"{self._external_knowledge}"
            )
        visual_signal = ""
        if self._get_visual_signal:
            try:
                visual_signal = (self._get_visual_signal() or "").strip()
            except Exception:
                visual_signal = ""
        if not visual_signal:
            return f"{self._omni_cfg.instructions}\n\n{self._omni_cfg.voice_instructions}{knowledge_block}"
        return (
            f"{self._omni_cfg.instructions}\n\n"
            f"{self._omni_cfg.voice_instructions}\n\n"
            "下面是本地视觉链路给出的结构化参考信号。它比原始视频更稳定，但也只是粗略参考；"
            "优先用它来调整语气、回应强度和节奏，不要逐字复述。\n"
            f"{visual_signal}"
            f"{knowledge_block}"
        )

    def _maybe_refresh_visual_signal(self) -> None:
        conv = self._conversation
        if conv is None or not self._running.is_set():
            return
        now = time.time()
        if now - self._last_signal_update_ts < 2.5:
            return
        instructions = self._build_runtime_instructions()
        if instructions == self._last_visual_signal:
            return
        try:
            conv.send_raw(
                json.dumps(
                    {
                        "event_id": f"event_local_visual_{int(now * 1000)}",
                        "type": "session.update",
                        "session": {"instructions": instructions},
                    }
                )
            )
            self._last_signal_update_ts = now
            self._last_visual_signal = instructions
        except Exception:
            pass

    def update_external_knowledge(self, snippets: str | None) -> None:
        cleaned = (snippets or "").strip()
        if cleaned == self._external_knowledge:
            return
        self._external_knowledge = cleaned
        self._last_signal_update_ts = 0.0
        self._maybe_refresh_visual_signal()
