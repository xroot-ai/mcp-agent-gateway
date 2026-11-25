"""ReAct-style orchestrator: the model reasons, calls tools, reads results, and answers.

Yields typed events as they happen. Text is passed through the marker parser so title
and follow-up markers never reach the client as raw text.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from gateway.agents.guardrails import BLOCK_MESSAGES
from gateway.agents.llm import LLM, ModelOutputError, ModelTurn
from gateway.agents.sources import previous_sources_context
from gateway.agents.stream_markers import MarkerParser
from gateway.api.schemas import (
    Progress,
    RunBlocked,
    RunCompleted,
    Source,
    TextDelta,
    TitleEvent,
)
from gateway.config.loader import Features, Limits
from gateway.tools.wrapper import ToolExecutor

AgentEvent = Progress | TextDelta | TitleEvent | RunCompleted | RunBlocked


@dataclass
class TurnInput:
    question: str
    history: list[dict[str, Any]] = field(default_factory=list)
    previous_sources: list[Source] = field(default_factory=list)
    want_title: bool = False


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value).split()) if value is not None else ""
    return text[:limit]


class Orchestrator:
    def __init__(
        self,
        llm: LLM,
        system_prompt: str,
        followups_prompt: str,
        features: Features,
        limits: Limits,
    ) -> None:
        self._llm = llm
        self._features = features
        self._limits = limits
        self._system = system_prompt
        if features.followups:
            self._system += "\n\n" + followups_prompt

    def _user_message(self, turn: TurnInput) -> str:
        parts = []
        if prev := previous_sources_context(turn.previous_sources):
            parts.append(f"<previous_sources>\n{prev}\n</previous_sources>")
        if turn.want_title and self._features.titles:
            parts.append("<title_requested/>")
        parts.append(f"<question>\n{turn.question}\n</question>")
        return "\n".join(parts)

    async def run(self, turn: TurnInput, executor: ToolExecutor) -> AsyncIterator[AgentEvent]:
        parser = MarkerParser()
        messages: list[dict[str, Any]] = [
            *turn.history,
            {"role": "user", "content": self._user_message(turn)},
        ]
        tools = executor.toolsets.definitions()
        emit_title = turn.want_title and self._features.titles

        for round_no in range(self._limits.max_tool_rounds + 1):
            allow_tools = round_no < self._limits.max_tool_rounds
            result: ModelTurn | None = None
            async for item in self._llm.stream(
                system=self._system, messages=messages, tools=tools, allow_tools=allow_tools
            ):
                if isinstance(item, ModelTurn):
                    result = item
                    continue
                parsed = parser.feed(item)
                if parsed.title and emit_title:
                    yield TitleEvent(title=parsed.title)
                if parsed.text:
                    yield TextDelta(text=parsed.text)
            if result is None:
                raise ModelOutputError("model stream ended without a final turn")

            if result.stop_reason == "refusal":
                yield RunBlocked(category="model_refusal", message=BLOCK_MESSAGES["model_refusal"])
                return
            if not result.tool_calls:
                break
            if result.stop_reason == "max_tokens":
                raise ModelOutputError("tool input was truncated by max_tokens")

            messages.append({"role": "assistant", "content": result.content})
            for call in result.tool_calls:
                binding = executor.toolsets.bindings.get(call.name)
                if binding and binding.policy.progress and isinstance(call.input, dict):
                    yield Progress(
                        tool=call.name,
                        title=_clip(call.input.get("progress_title"), 80) or "Working",
                        detail=_clip(call.input.get("progress_detail"), 200),
                    )
            outcomes = await asyncio.gather(
                *(executor.execute(c.name, c.input) for c in result.tool_calls)
            )
            # All results go back in one user message (keeps parallel tool use working).
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": c.id,
                            "content": o.content,
                            "is_error": o.is_error,
                        }
                        for c, o in zip(result.tool_calls, outcomes, strict=True)
                    ],
                }
            )

        tail = parser.finish()
        if tail.text:
            yield TextDelta(text=tail.text)
        answer = parser.answer
        yield RunCompleted(
            answer=answer,
            sources=executor.sources.cited_in(answer),
            followups=parser.followups if self._features.followups else [],
            title=parser.title if emit_title else None,
        )
