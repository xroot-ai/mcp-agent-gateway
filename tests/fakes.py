"""Test doubles: a scripted model, a scripted guardrails classifier, a canned searcher."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any

from gateway.agents.guardrails import Verdict
from gateway.agents.llm import ModelTurn, ToolCall
from mcp_servers.retrieval.overrides import SearchOverrides
from mcp_servers.retrieval.search import Passage


@dataclass
class Script:
    text: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    delay: float = 0.0


class FakeLLM:
    def __init__(self, scripts: list[Script]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []

    async def stream(self, *, system, messages, tools, allow_tools):  # noqa: ANN001
        self.calls.append(
            {
                "system": system,
                "messages": copy.deepcopy(messages),
                "tools": tools,
                "allow_tools": allow_tools,
            }
        )
        script = self.scripts.pop(0) if self.scripts else Script(text=["(no script left)"])
        for chunk in script.text:
            if script.delay:
                await asyncio.sleep(script.delay)
            yield chunk
        content: list[dict[str, Any]] = []
        if script.text:
            content.append({"type": "text", "text": "".join(script.text)})
        content += [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
            for c in script.tool_calls
        ]
        stop = script.stop_reason or ("tool_use" if script.tool_calls else "end_turn")
        yield ModelTurn(content=content, stop_reason=stop, tool_calls=list(script.tool_calls))


class FakeGuard:
    def __init__(
        self, verdict: Verdict | None = None, delay: float = 0.0, error: Exception | None = None
    ) -> None:
        self.verdict = verdict or Verdict(allowed=True, category="allowed", reason="ok")
        self.delay = delay
        self.error = error
        self.seen: list[str] = []

    async def classify(self, message: str, recent_questions: list[str]) -> Verdict:
        self.seen.append(message)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.verdict


BLOCK = Verdict(allowed=False, category="prompt_injection", reason="tries to override rules")


class FakeSearcher:
    def __init__(self, passages: list[Passage] | None = None) -> None:
        self.passages = (
            passages
            if passages is not None
            else [
                Passage(
                    "Pod Lifecycle",
                    "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
                    "Container restart policy",
                    "The restartPolicy applies to all containers "
                    "in the Pod. Possible values are Always, OnFailure, and Never.",
                    0.91,
                ),
                Passage(
                    "Jobs",
                    "https://kubernetes.io/docs/concepts/workloads/controllers/job/",
                    "Pod backoff failure policy",
                    "A Job's Pods are retried up to backoffLimit times, default 6.",
                    0.77,
                ),
            ]
        )
        self.queries: list[tuple[str, int, SearchOverrides]] = []

    async def search(self, query: str, top_k: int, overrides: SearchOverrides) -> list[Passage]:
        self.queries.append((query, top_k, overrides))
        return self.passages[:top_k]


def search_call(call_id: str = "t1", query: str = "pod restart policy", **extra: Any) -> ToolCall:
    return ToolCall(
        call_id,
        "search",
        {
            "query": query,
            "progress_title": "Searching pod docs",
            "progress_detail": "Looking up restart policy",
            **extra,
        },
    )
