"""Offline demo mode (``GW_OFFLINE_LLM=true``): run the gateway without an Anthropic API key.

``OfflineLLM`` stands in for the orchestrator model. It doesn't write answers: it runs one
real ``search`` with the user's question, then streams back the top passages it got, with
real ``[n]`` citations, a title and follow-ups. ``OfflineGuardrails`` stands in for the
classifier with a few keyword rules. Everything around them is the real pipeline: SSE,
progress events, the guardrails gate, MCP tool calls, source numbering, persistence and
cancellation.

This is for demos and local smoke tests only. Answers are retrieved text, not model output.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from gateway.agents.guardrails import Verdict
from gateway.agents.llm import ModelTurn, ToolCall

NOTICE = (
    "_Offline demo mode: no model is connected, so below are the most relevant passages "
    "from the docs rather than a written answer._\n\n"
)
_QUESTION = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
_HEADING = re.compile(r"^\[(\d+)\] (.+)$", re.MULTILINE)
_TICKET = re.compile(r"\b(open|create|file|raise)\b.{0,20}\bticket\b", re.IGNORECASE)


def _last_question(messages: list[dict[str, Any]]) -> tuple[str, bool]:
    for msg in reversed(messages):
        if msg["role"] == "user" and isinstance(msg["content"], str):
            found = _QUESTION.findall(msg["content"])
            question = found[-1] if found else msg["content"]
            return question.strip(), "<title_requested/>" in msg["content"]
    return "", False


def _first_sentences(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return (cut[: end + 1] if end > limit // 2 else cut.rstrip()) + " …"


def _passages(tool_text: str) -> list[tuple[int, str, str]]:
    """Parse ``render_for_model`` output back into (number, heading, text)."""
    marks = list(_HEADING.finditer(tool_text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(tool_text)
        out.append((int(m.group(1)), m.group(2).strip(), tool_text[m.end() : end].strip()))
    return out


def _title(question: str) -> str:
    words = re.sub(r"[^\w\s-]", "", question).split()
    title = " ".join(words[:6])
    return (title[:1].upper() + title[1:]) or "Kubernetes question"


class OfflineLLM:
    """Implements the ``LLM`` protocol with retrieval instead of generation."""

    def __init__(self, chunk_delay_s: float = 0.02, max_passages: int = 3) -> None:
        self._delay = chunk_delay_s
        self._max = max_passages

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        allow_tools: bool,
    ) -> AsyncIterator[str | ModelTurn]:
        question, want_title = _last_question(messages)
        tool_names = {t["name"] for t in tools}
        last = messages[-1]
        first_round = isinstance(last["content"], str)

        if first_round and allow_tools:
            call = self._plan(question, tool_names)
            if call is not None:
                content = [
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                ]
                yield ModelTurn(content=content, stop_reason="tool_use", tool_calls=[call])
                return

        text = self._compose(question, want_title, last if not first_round else None)
        for chunk in re.findall(r"\S+\s*|\s+", text):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield chunk
        yield ModelTurn(
            content=[{"type": "text", "text": text}], stop_reason="end_turn", tool_calls=[]
        )

    @staticmethod
    def _plan(question: str, tool_names: set[str]) -> ToolCall | None:
        if _TICKET.search(question) and "create_ticket" in tool_names:
            return ToolCall(
                "offline_ticket",
                "create_ticket",
                {
                    "title": (question[:117] + "...") if len(question) > 120 else question.ljust(5),
                    "description": question.ljust(10),
                    "priority": "normal",
                },
            )
        if "search" in tool_names and len(question) >= 2:
            return ToolCall(
                "offline_search",
                "search",
                {
                    "query": question[:300],
                    "progress_title": "Searching the Kubernetes docs",
                    "progress_detail": f"Looking for: {question[:150]}",
                    "top_k": 5,
                },
            )
        return None

    def _compose(self, question: str, want_title: bool, results: dict[str, Any] | None) -> str:
        parts = [f"[[TITLE: {_title(question)}]]\n"] if want_title else []
        blocks = results["content"] if results and isinstance(results["content"], list) else []
        result = next((b for b in blocks if b.get("type") == "tool_result"), None)
        tool_text = str(result.get("content", "")) if result else ""

        if result and result["tool_use_id"] == "offline_ticket":
            status = "could not be created" if result.get("is_error") else "was created"
            try:
                ticket = json.loads(tool_text)
                detail = f"Ticket **{ticket['ticket_id']}** is {ticket['status']}."
            except (ValueError, KeyError, TypeError):
                detail = tool_text
            parts.append(f"Offline demo mode: the support ticket {status}.\n\n{detail}")
            return "".join(parts)

        passages = _passages(tool_text)[: self._max]
        if not passages:
            parts.append(NOTICE + "No matching passages were found in the documentation.")
            return "".join(parts)

        parts.append(NOTICE)
        for number, heading, text in passages:
            parts.append(f"**{heading}**: {_first_sentences(text)} [{number}]\n\n")
        pages = []
        for _, heading, _ in passages:
            page = heading.split(" > ")[0]
            if page not in pages:
                pages.append(page)
        parts.append("[[FOLLOWUPS]]\n" + "".join(f"- What else covers {p}?\n" for p in pages))
        return "".join(parts)


_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "prompt_injection",
        re.compile(
            r"ignore (all |any |your |the |previous |prior )+(instructions|rules)"
            r"|(reveal|print|show|repeat) (me )?(your|the) (system )?prompt"
            r"|you are now|developer mode|jailbreak|\bDAN\b",
            re.IGNORECASE,
        ),
    ),
    (
        "harmful",
        re.compile(r"\b(malware|ransomware|ddos|exploit (a|the) cluster)\b", re.IGNORECASE),
    ),
]


class OfflineGuardrails:
    """Implements the ``Guardrails`` protocol with keyword rules. Demo use only."""

    def __init__(self, delay_s: float = 0.3) -> None:
        self._delay = delay_s  # mimics classifier latency so the gate's buffering is visible

    async def classify(self, message: str, recent_questions: list[str]) -> Verdict:
        if self._delay:
            await asyncio.sleep(self._delay)
        for category, pattern in _RULES:
            if pattern.search(message):
                return Verdict(allowed=False, category=category, reason="offline keyword rule")  # type: ignore[arg-type]
        return Verdict(allowed=True, category="allowed", reason="offline keyword rules passed")
