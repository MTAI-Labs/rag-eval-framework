"""Chat client for the judge panel.

All three judges are served by the Thoth gateway and called through the
standard ``openai`` SDK against ``${THOTH_BASE_URL}/v1`` (design spec §5.2).

The panel depends on the :class:`ChatClient` protocol rather than on the SDK,
so tests inject a scripted client and the CI smoke run needs no gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str = ""


class ChatClient(Protocol):
    """Minimal chat-completions surface the judge panel needs."""

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse: ...


class ThothChatClient:
    """OpenAI-SDK client pointed at the Thoth gateway."""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 120.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "the judge panel needs the openai SDK: pip install -e '.[judge]'"
            ) from exc
        self.base_url = base_url
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        usage = getattr(response, "usage", None)
        return ChatResponse(
            text=(response.choices[0].message.content or "") if response.choices else "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            model=getattr(response, "model", model),
        )


class ScriptedChatClient:
    """Deterministic client for tests and the offline smoke run.

    ``responses`` maps a model id to either a fixed JSON string or a callable
    ``(messages) -> str``. Unknown models fall back to ``default``.
    """

    def __init__(self, responses: dict[str, Any], default: Any = None) -> None:
        self.responses = responses
        self.default = default
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        self.calls.append((model, messages))
        handler = self.responses.get(model, self.default)
        if handler is None:
            raise RuntimeError(f"ScriptedChatClient has no response for model {model!r}")
        text = handler(messages) if callable(handler) else str(handler)
        return ChatResponse(text=text, prompt_tokens=len(str(messages)) // 4,
                            completion_tokens=len(text) // 4, model=model)


def build_client(config: Any) -> ChatClient:
    """Construct the real gateway client from a :class:`JudgeConfig`."""
    return ThothChatClient(config.base_url, config.api_key, timeout=config.timeout_s)
