from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, Any

from ..config import DashScopeConfig, RealtimeAsrConfig


class RealtimeAsrUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class AsrEvent:
    text: str
    is_final: bool = False


class RealtimeAsr:
    """
    基于 DashScope SDK 的实时语音识别（WebSocket）。

    说明：
    - SDK 的示例使用 pyaudio；这里默认用 sounddevice（更易在 Windows 上安装/使用）。
    - 如环境缺少 sounddevice，可改用 pyaudio 或关闭语音功能。
    """

    def __init__(self, *, dashscope: DashScopeConfig, asr: RealtimeAsrConfig):
        self._dashscope_cfg = dashscope
        self._asr_cfg = asr

        self._translator: Any = None
        self._stream: Any = None
        self._running = threading.Event()
        self._lock = threading.Lock()

    def start(self, on_event: Callable[[AsrEvent], None]) -> None:
        with self._lock:
            if self._running.is_set():
                return
            self._running.set()

        try:
            import dashscope  # type: ignore
            from dashscope.audio.asr import TranslationRecognizerRealtime, TranslationRecognizerCallback  # type: ignore
        except ImportError as exc:
            self._running.clear()
            raise RealtimeAsrUnavailableError(
                "dashscope 或其实时ASR组件不可用。请先执行: pip install dashscope"
            ) from exc

        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            self._running.clear()
            raise RealtimeAsrUnavailableError(
                "sounddevice 未安装。请先执行: pip install sounddevice"
            ) from exc

        dashscope.api_key = self._dashscope_cfg.api_key
        if self._dashscope_cfg.base_http_api_url:
            dashscope.base_http_api_url = self._dashscope_cfg.base_http_api_url

        running_flag = self._running

        class _Cb(TranslationRecognizerCallback):  # type: ignore[misc]
            def on_open(self) -> None:
                pass

            def on_close(self) -> None:
                pass

            def on_event(self, request_id, transcription_result, translation_result, usage) -> None:  # noqa: ANN001
                if not running_flag.is_set():
                    return
                tr = transcription_result
                if tr is None:
                    return
                text = getattr(tr, "text", None)
                if not text:
                    return
                is_final = bool(
                    getattr(tr, "is_final", False)
                    or getattr(tr, "sentence_end", False)
                    or getattr(tr, "end", False)
                )
                on_event(AsrEvent(text=str(text), is_final=is_final))

        callback = _Cb()

        translator = TranslationRecognizerRealtime(
            model=self._asr_cfg.model,
            format="pcm",
            sample_rate=self._asr_cfg.sample_rate,
            transcription_enabled=True,
            translation_enabled=False,
            callback=callback,
        )
        translator.start()

        def _audio_cb(indata, frames, time_info, status):  # noqa: ANN001
            if not running_flag.is_set():
                return
            try:
                translator.send_audio_frame(bytes(indata))
            except Exception:
                pass

        stream = sd.RawInputStream(
            samplerate=self._asr_cfg.sample_rate,
            blocksize=int(self._asr_cfg.sample_rate / 10),  # ~100ms
            channels=1,
            dtype="int16",
            callback=_audio_cb,
        )
        stream.start()

        with self._lock:
            self._translator = translator
            self._stream = stream

    def stop(self) -> None:
        with self._lock:
            if not self._running.is_set():
                return
            self._running.clear()
            stream = self._stream
            translator = self._translator
            self._stream = None
            self._translator = None

        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:
            pass
        try:
            if translator is not None:
                translator.stop()
        except Exception:
            pass
