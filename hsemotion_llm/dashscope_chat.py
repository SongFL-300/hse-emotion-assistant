from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Iterable, Iterator

from .config import DashScopeConfig


class DashScopeNotInstalledError(ImportError):
    pass


@dataclass(frozen=True)
class StreamChunk:
    text: str
    finish_reason: str | None
    raw: Any


class DashScopeChatClient:
    """
    DashScope/Qwen 的流式对话客户端（多轮 messages 输入，流式增量输出）。
    """

    def __init__(self, config: DashScopeConfig):
        try:
            import dashscope  # type: ignore
        except ImportError as exc:
            raise DashScopeNotInstalledError(
                "dashscope 未安装。请先执行: pip install dashscope"
            ) from exc

        from dashscope import Generation  # type: ignore
        from dashscope import MultiModalConversation  # type: ignore

        self._dashscope = dashscope
        self._Generation = Generation
        self._MultiModalConversation = MultiModalConversation

        self._dashscope.api_key = config.api_key
        self._model = config.model
        self._incremental_output = config.incremental_output
        self._enable_thinking = bool(config.enable_thinking)
        self._thinking_budget = int(config.thinking_budget)
        self._use_multimodal_endpoint = self._should_use_multimodal_endpoint(self._model)
        if config.base_http_api_url:
            self._dashscope.base_http_api_url = config.base_http_api_url
        elif self._use_multimodal_endpoint:
            self._dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = 0.7,
        top_p: float | None = 0.8,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[StreamChunk]:
        if self._use_multimodal_endpoint:
            yield from self._stream_chat_multimodal(
                messages,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_tokens=max_tokens,
            )
            return

        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "result_format": "message",
            "stream": True,
            "incremental_output": self._incremental_output,
        }
        params["enable_thinking"] = self._enable_thinking
        if self._enable_thinking and self._thinking_budget > 0:
            params["thinking_budget"] = min(1999, int(self._thinking_budget))
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if seed is not None:
            params["seed"] = seed
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        responses: Iterable[Any] = self._Generation.call(**params)
        for resp in responses:
            if getattr(resp, "status_code", None) != HTTPStatus.OK:
                raise RuntimeError(
                    f"DashScope 请求失败: request_id={getattr(resp, 'request_id', None)} "
                    f"code={getattr(resp, 'code', None)} message={getattr(resp, 'message', None)}"
                )

            try:
                choice = resp.output.choices[0]
                content = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
            except Exception:
                content = ""
                finish_reason = None

            yield StreamChunk(text=content, finish_reason=finish_reason, raw=resp)

    def _stream_chat_multimodal(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        top_p: float | None,
        seed: int | None,
        max_tokens: int | None,
    ) -> Iterator[StreamChunk]:
        params: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_multimodal_message(m) for m in messages],
            "stream": True,
            "enable_thinking": self._enable_thinking,
        }
        if self._enable_thinking and self._thinking_budget > 0:
            params["thinking_budget"] = min(1999, int(self._thinking_budget))
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if seed is not None:
            params["seed"] = seed
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        responses: Iterable[Any] = self._MultiModalConversation.call(**params)
        for resp in responses:
            if getattr(resp, "status_code", None) != HTTPStatus.OK:
                raise RuntimeError(
                    f"DashScope 请求失败: request_id={getattr(resp, 'request_id', None)} "
                    f"code={getattr(resp, 'code', None)} message={getattr(resp, 'message', None)}"
                )

            try:
                choice = resp.output.choices[0]
                raw_content = choice.message.content or []
                content = self._extract_multimodal_text(raw_content)
                if not content:
                    content = self._extract_multimodal_text(getattr(resp.output, "text", None))
                finish_reason = getattr(choice, "finish_reason", None)
            except Exception:
                content = ""
                finish_reason = None

            yield StreamChunk(text=content, finish_reason=finish_reason, raw=resp)

    def _to_multimodal_message(self, message: dict[str, str]) -> dict[str, Any]:
        return {
            "role": message.get("role", "user"),
            "content": [{"text": str(message.get("content", ""))}],
        }

    def _extract_multimodal_text(self, raw_content: Any) -> str:
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            return "".join(parts)
        return str(raw_content or "")

    def _should_use_multimodal_endpoint(self, model: str) -> bool:
        name = (model or "").strip().lower()
        return name.startswith("qwen3.5-")

    def set_runtime_options(
        self,
        *,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        if enable_thinking is not None:
            self._enable_thinking = bool(enable_thinking)
        if thinking_budget is not None:
            self._thinking_budget = min(1999, max(0, int(thinking_budget)))
