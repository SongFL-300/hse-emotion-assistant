from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import asyncio
import base64
import json
import threading
import time
from typing import Callable, Any

from ..config import DashScopeConfig, RealtimeTtsConfig


class SessionMode(Enum):
    SERVER_COMMIT = "server_commit"
    COMMIT = "commit"


@dataclass(frozen=True)
class TtsEvent:
    type: str
    payload: dict[str, Any]


class RealtimeTtsUnavailableError(RuntimeError):
    pass


class TtsTextChunker:
    """
    将 LLM 的增量输出拆成更适合 TTS 的小片段，避免每个 token 都发一条 WebSocket 事件。
    """

    def __init__(self, *, min_chars: int = 12, max_chars: int = 80):
        self._min = int(min_chars)
        self._max = int(max_chars)
        self._buf = ""

    def push(self, s: str) -> list[str]:
        if not s:
            return []
        self._buf += s
        out: list[str] = []

        def _flush(n: int) -> None:
            nonlocal out
            chunk = self._buf[:n]
            self._buf = self._buf[n:]
            if chunk.strip():
                out.append(chunk)

        # 优先按中文/英文句末标点切
        while True:
            if len(self._buf) < self._min:
                break
            cut = -1
            for p in ("。", "！", "？", ".", "!", "?", "；", ";", "\n"):
                idx = self._buf.find(p)
                if idx != -1 and idx + 1 >= self._min:
                    cut = idx + 1
                    break
            if cut != -1:
                _flush(cut)
                continue
            if len(self._buf) >= self._max:
                _flush(self._max)
                continue
            break

        return out

    def flush(self) -> list[str]:
        if not self._buf.strip():
            self._buf = ""
            return []
        s = self._buf
        self._buf = ""
        return [s]


class RealtimeTts:
    """
    Qwen TTS Realtime WebSocket 客户端（参考官方文档实现，低延时流式音频输出）。
    """

    def __init__(
        self,
        *,
        dashscope: DashScopeConfig,
        tts: RealtimeTtsConfig,
        audio_callback: Callable[[bytes], None] | None = None,
        event_callback: Callable[[TtsEvent], None] | None = None,
    ):
        self._dashscope_cfg = dashscope
        self._tts_cfg = tts
        self._audio_callback = audio_callback
        self._event_callback = event_callback

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: Any = None
        self._response_done: asyncio.Future[bool] | None = None

        self._connected = threading.Event()
        self._closing = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._closing.clear()
        self._connected.clear()
        self._thread = threading.Thread(target=self._run_loop, name="RealtimeTTS", daemon=True)
        self._thread.start()
        # 等待连接建立（给一点时间，避免 UI 卡住）
        self._connected.wait(timeout=3.0)

    def stop(self) -> None:
        self._closing.set()
        loop = self._loop
        if loop:
            asyncio.run_coroutine_threadsafe(self._close_ws(), loop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._thread = None
        self._loop = None

    def append_text(self, text: str) -> None:
        if not text or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._append_text(text), self._loop)

    def finish(self) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._finish_session(), self._loop)

    def wait_done(self, timeout_s: float = 10.0) -> bool:
        loop = self._loop
        fut = self._response_done
        if not loop or not fut:
            return True
        try:
            return bool(asyncio.run_coroutine_threadsafe(asyncio.wait_for(fut, timeout=timeout_s), loop).result())
        except Exception:
            return False

    def _run_loop(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())
        except Exception:
            pass
        finally:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.close()
            except Exception:
                pass

    async def _main(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError as exc:
            raise RealtimeTtsUnavailableError(
                "websockets 未安装。请先执行: pip install websockets"
            ) from exc

        headers = {"Authorization": f"Bearer {self._dashscope_cfg.api_key}"}
        self._ws = await websockets.connect(self._tts_cfg.base_ws_url, additional_headers=headers)

        await self._send_event(
            {
                "type": "session.update",
                "session": {
                    "mode": SessionMode.SERVER_COMMIT.value,
                    "voice": self._tts_cfg.voice,
                    "language_type": self._tts_cfg.language_type,
                    "response_format": "pcm",
                    "sample_rate": self._tts_cfg.sample_rate,
                },
            }
        )

        self._response_done = asyncio.Future()
        self._connected.set()

        consumer = asyncio.create_task(self._handle_messages())
        await consumer

    async def _send_event(self, event: dict[str, Any]) -> None:
        if not self._ws:
            return
        event = dict(event)
        event["event_id"] = "event_" + str(int(time.time() * 1000))
        await self._ws.send(json.dumps(event, ensure_ascii=False))

    async def _append_text(self, text: str) -> None:
        await self._send_event({"type": "input_text_buffer.append", "text": text})

    async def _finish_session(self) -> None:
        await self._send_event({"type": "session.finish"})

    async def _close_ws(self) -> None:
        ws = self._ws
        self._ws = None
        try:
            if ws is not None:
                await ws.close()
        except Exception:
            pass

    async def _handle_messages(self) -> None:
        if not self._ws:
            return
        try:
            async for message in self._ws:
                event = json.loads(message)
                event_type = str(event.get("type", ""))

                if self._event_callback:
                    self._event_callback(TtsEvent(type=event_type, payload=event))

                if event_type == "response.audio.delta" and self._audio_callback:
                    audio_bytes = base64.b64decode(event.get("delta", "") or "")
                    if audio_bytes:
                        self._audio_callback(audio_bytes)
                elif event_type == "response.done":
                    if self._response_done and not self._response_done.done():
                        self._response_done.set_result(True)
                elif event_type == "error":
                    if self._response_done and not self._response_done.done():
                        self._response_done.set_result(False)

                if self._closing.is_set():
                    break
        except Exception:
            if self._response_done and not self._response_done.done():
                self._response_done.set_result(False)
        finally:
            await self._close_ws()
