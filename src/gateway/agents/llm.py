"""Model access behind a small streaming interface, so tests can script a fake model."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic import AsyncAnthropic

from gateway.config.loader import ModelSpec
from gateway.observability import metrics
from gateway.observability.tracing import span

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ModelOutputError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    input: Any


@dataclass
class ModelTurn:
    content: list[Any]
    stop_reason: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        allow_tools: bool,
    ) -> AsyncIterator[str | ModelTurn]:
        """Yield text deltas, then exactly one ``ModelTurn``."""
        ...


class AnthropicLLM:
    def __init__(
        self,
        spec: ModelSpec,
        fallbacks: str | None = "default",
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._spec = spec
        self._fallbacks = fallbacks
        self._client = client or AsyncAnthropic()

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        allow_tools: bool,
    ) -> AsyncIterator[str | ModelTurn]:
        kwargs: dict[str, Any] = {
            "model": self._spec.model,
            "max_tokens": self._spec.max_tokens,
            # Stable system prompt first so the prefix caches across requests.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._spec.effort},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "auto" if allow_tools else "none"}
        if self._fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = self._fallbacks

        started = time.perf_counter()
        attempts = 0
        with span("llm.turn", model=self._spec.model) as s:
            while True:
                emitted = False
                try:
                    async with self._client.beta.messages.stream(**kwargs) as stream:
                        async for event in stream:
                            if event.type == "text":
                                emitted = True
                                yield event.text
                        message = await stream.get_final_message()
                    break
                except ValueError as exc:
                    # Tool input JSON the SDK could not parse at all (eager input streaming).
                    attempts += 1
                    if emitted or attempts > 2:
                        raise ModelOutputError("model produced unparseable tool input") from exc
                    log.warning("unparseable tool input; re-issuing turn (attempt %d)", attempts)

            usage = message.usage
            s.set_attribute("llm.stop_reason", str(message.stop_reason))
            s.set_attribute("llm.served_by", message.model)
            s.set_attribute("llm.input_tokens", usage.input_tokens)
            s.set_attribute("llm.output_tokens", usage.output_tokens)
            s.set_attribute("llm.cache_read_tokens", usage.cache_read_input_tokens or 0)
            metrics.llm_calls.add(1, {"model": self._spec.model, "stop": str(message.stop_reason)})
            metrics.llm_latency.record(time.perf_counter() - started, {"model": self._spec.model})
            metrics.llm_tokens.add(usage.input_tokens, {"direction": "input"})
            metrics.llm_tokens.add(usage.output_tokens, {"direction": "output"})

        calls = [ToolCall(b.id, b.name, b.input) for b in message.content if b.type == "tool_use"]
        yield ModelTurn(
            content=list(message.content), stop_reason=message.stop_reason, tool_calls=calls
        )
