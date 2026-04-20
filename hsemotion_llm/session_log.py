from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any

from .config import AppConfig


@dataclass(frozen=True)
class LogPaths:
    jsonl: Path
    md: Path


class ChatSessionLogger:
    """
    Local session logger + history writer.

    - Writes structured JSONL (easy to parse/restore in UI)
    - Writes a readable Markdown transcript with extra metadata
    """

    def __init__(self, *, config: AppConfig, base_dir: Path | None = None):
        self._config = config
        self._lock = threading.Lock()

        base = base_dir or Path(".hsemotion_logs")
        base.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = ts
        self.paths = LogPaths(
            jsonl=base / f"chat_{ts}.jsonl",
            md=base / f"chat_{ts}.md",
        )

        self._write_md_header()
        self.event("session_start", model=config.dashscope.model)

    def event(self, event_type: str, **payload: Any) -> None:
        rec = {"ts": time.time(), "type": event_type, "payload": payload}
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self.paths.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.paths.jsonl.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_message(self, *, role: str, text: str, meta: dict[str, Any] | None = None) -> None:
        self.event("message", role=role, text=text, meta=meta or {})
        self._append_md_message(role=role, text=text, meta=meta or {})

    def log_emotion_sample(self, *, play_s: float, emotion_summary: str | None) -> None:
        self.event("emotion_sample", play_s=float(play_s), emotion_summary=emotion_summary or "")

    def _write_md_header(self) -> None:
        with self._lock:
            with self.paths.md.open("w", encoding="utf-8") as f:
                f.write(f"# Chat Session {self.session_id}\n\n")
                f.write("## Config\n")
                f.write(f"- model: `{self._config.dashscope.model}`\n")
                f.write(f"- asr_enabled: `{bool(self._config.asr.enabled)}`\n")
                f.write(f"- tts_enabled: `{bool(self._config.tts.enabled)}`\n")
                f.write(f"- rag_enabled(default): `{bool(self._config.rag.enabled)}`\n")
                f.write("\n---\n\n")

    def _append_md_message(self, *, role: str, text: str, meta: dict[str, Any]) -> None:
        title = "User" if role == "user" else "Assistant" if role == "assistant" else role
        with self._lock:
            with self.paths.md.open("a", encoding="utf-8") as f:
                f.write(f"## {title}\n\n")
                f.write(text.strip() + "\n\n")
                if meta:
                    f.write("<details><summary>meta</summary>\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(meta, ensure_ascii=False, indent=2))
                    f.write("\n```\n\n</details>\n\n")

