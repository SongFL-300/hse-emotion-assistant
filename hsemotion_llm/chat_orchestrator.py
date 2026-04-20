from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .config import AppConfig
from .conversation import ConversationSession
from .dashscope_chat import DashScopeChatClient
from .prompts import (
    BASE_SYSTEM_PROMPT,
    build_dynamic_context,
    build_emotion_context,
    build_emotion_trace_context,
    build_rag_context,
)


@dataclass(frozen=True)
class ChatHooks:
    get_emotion_summary: Callable[[], str | None] | None = None
    get_emotion_trace: Callable[[], str | None] | None = None
    get_emotion_status: Callable[[], Any] | None = None
    get_rag_snippets: Callable[[str], str | None] | None = None
    get_rag_snippets_with_meta: Callable[[str], tuple[str | None, int, list[str]]] | None = None


class EmotionAwareChat:
    def __init__(
        self,
        config: AppConfig,
        *,
        hooks: ChatHooks | None = None,
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ):
        self._config = config
        self._client = DashScopeChatClient(config.dashscope)
        self._session = ConversationSession(system_prompt=system_prompt)
        self._hooks = hooks or ChatHooks()
        self.last_reply_meta: dict[str, Any] = {}

    @property
    def session(self) -> ConversationSession:
        return self._session

    def reset(self) -> None:
        self._session.reset()

    def stream_reply(
        self,
        user_text: str,
        *,
        realtime_adjust: bool = False,
        adjust_threshold: float = 0.45,
        max_tokens_per_segment: int = 160,
        max_segments: int = 4,
    ) -> Iterator[str]:
        """
        流式生成回复。

        - `realtime_adjust=False`：单次请求，情绪只在“请求开始时”注入一次（更省 token）。
        - `realtime_adjust=True`：仅当“最近几秒情绪变化幅度 >= 阈值”时，才启用分段请求（更自然，减少打断）。
        """
        emotion_summary = self._hooks.get_emotion_summary() if self._hooks.get_emotion_summary else None
        emotion_trace = self._hooks.get_emotion_trace() if self._hooks.get_emotion_trace else None
        rag_snippets: str | None = None
        rag_hit_count = 0
        rag_sources: list[str] = []
        rag_retrieval_triggered = False
        if self._hooks.get_rag_snippets_with_meta:
            rag_snippets, rag_hit_count, rag_sources = self._hooks.get_rag_snippets_with_meta(user_text)
            rag_retrieval_triggered = bool(rag_snippets) or rag_hit_count > 0
        elif self._hooks.get_rag_snippets:
            rag_snippets = self._hooks.get_rag_snippets(user_text)
            rag_retrieval_triggered = bool(rag_snippets)

        emotion_change_score: float | None = None
        if self._hooks.get_emotion_status:
            try:
                status = self._hooks.get_emotion_status()
                timeline = getattr(status, "timeline", None)
                if timeline is not None and hasattr(timeline, "change_score"):
                    emotion_change_score = timeline.change_score()
            except Exception:
                emotion_change_score = None

        should_segment = bool(realtime_adjust) and (
            emotion_change_score is not None and float(emotion_change_score) >= float(adjust_threshold)
        )

        emotion_ctx_preview = build_emotion_context(emotion_summary)
        trace_ctx_preview = build_emotion_trace_context(emotion_trace)
        self.last_reply_meta = {
            "realtime_adjust_enabled": bool(realtime_adjust),
            "realtime_adjust_triggered": bool(should_segment),
            "emotion_change_score": emotion_change_score,
            "adjust_threshold": float(adjust_threshold),
            "max_tokens_per_segment": int(max_tokens_per_segment),
            "max_segments": int(max_segments),
            "emotion_summary_present": bool(emotion_summary),
            "emotion_context_injected": bool(emotion_ctx_preview),
            "emotion_trace_present": bool(emotion_trace),
            "emotion_trace_injected": bool(trace_ctx_preview),
            "rag_snippets_present": bool(rag_snippets),
            "rag_retrieval_triggered": bool(rag_retrieval_triggered),
            "rag_hit_count": int(rag_hit_count),
            "rag_sources": list(rag_sources),
            "rag_injected": bool(build_rag_context(rag_snippets)),
        }

        if not should_segment:
            dynamic_context = build_dynamic_context(
                emotion_summary=emotion_summary,
                emotion_trace=emotion_trace,
                rag_snippets=rag_snippets,
            )

            messages = self._session.build_messages(
                user_text=user_text,
                dynamic_system_context=dynamic_context if dynamic_context else None,
            )

            parts: list[str] = []
            for chunk in self._client.stream_chat(messages):
                if chunk.text:
                    parts.append(chunk.text)
                    yield chunk.text
                if chunk.finish_reason == "stop":
                    break

            assistant_text = "".join(parts).strip()
            self._session.add_user(user_text)
            self._session.add_assistant(assistant_text)
            return

        # should_segment=True：RAG 只注入一次；情绪每段刷新一次
        rag_ctx = build_rag_context(rag_snippets)
        trace_ctx = build_emotion_trace_context(emotion_trace)

        assistant_parts: list[str] = []
        is_done = False

        for seg in range(max(1, int(max_segments))):
            emotion_summary = self._hooks.get_emotion_summary() if self._hooks.get_emotion_summary else None
            emotion_ctx = build_emotion_context(emotion_summary)

            messages: list[dict[str, str]] = [{"role": "system", "content": self._session.system_prompt}]
            messages.extend(self._session.history)

            if rag_ctx:
                messages.append({"role": "system", "content": rag_ctx})
            if trace_ctx:
                messages.append({"role": "system", "content": trace_ctx})
            if emotion_ctx:
                messages.append({"role": "system", "content": emotion_ctx})

            messages.append({"role": "user", "content": user_text})

            if assistant_parts:
                messages.append({"role": "assistant", "content": "".join(assistant_parts)})
                messages.append(
                    {
                        "role": "user",
                        "content": "继续你的上一次回答（不要重复前文）。",
                    }
                )

            segment_has_text = False
            last_finish: str | None = None
            for chunk in self._client.stream_chat(messages, max_tokens=int(max_tokens_per_segment)):
                if chunk.text:
                    segment_has_text = True
                    assistant_parts.append(chunk.text)
                    yield chunk.text
                last_finish = chunk.finish_reason
                if chunk.finish_reason == "stop":
                    is_done = True
                    break

            if is_done:
                break
            if not segment_has_text:
                # 防止空输出死循环
                break
            # 如果模型因为长度截断，下一段继续；否则保守结束
            if last_finish and last_finish not in {"length", "stop"}:
                break

        assistant_text = "".join(assistant_parts).strip()
        self._session.add_user(user_text)
        self._session.add_assistant(assistant_text)
