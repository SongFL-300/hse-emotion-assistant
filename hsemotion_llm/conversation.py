from __future__ import annotations

from dataclasses import dataclass, field


Message = dict[str, str]


@dataclass
class ConversationSession:
    """
    仅存储多轮对话历史（user/assistant），动态上下文（情绪/RAG）不入库，避免污染记忆。
    """

    system_prompt: str
    max_history_messages: int = 24
    history: list[Message] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def build_messages(
        self,
        *,
        user_text: str,
        dynamic_system_context: str | None = None,
    ) -> list[Message]:
        messages: list[Message] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        if dynamic_system_context:
            messages.append({"role": "system", "content": dynamic_system_context})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _trim(self) -> None:
        if self.max_history_messages <= 0:
            self.history.clear()
            return
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages :]

