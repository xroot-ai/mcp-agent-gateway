"""Guardrails agent: classifies the incoming message into a structured verdict."""

from __future__ import annotations

import logging
from typing import Literal, Protocol

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from gateway.config.loader import ModelSpec

log = logging.getLogger(__name__)

Category = Literal[
    "allowed",
    "prompt_injection",
    "harmful",
    "sensitive_data",
    "off_topic",
    "guardrails_unavailable",
]

BLOCK_MESSAGES: dict[str, str] = {
    "prompt_injection": "I can't follow instructions that try to change how this assistant works.",
    "harmful": "I can't help with that request.",
    "sensitive_data": "I can't help with requests for credentials or other people's private data.",
    "off_topic": "I can only help with questions about the documentation this assistant covers.",
    "guardrails_unavailable": "The safety check is unavailable right now. Please try again.",
    "model_refusal": "I can't help with that request.",
}


class Verdict(BaseModel):
    allowed: bool = Field(description="True if the assistant may answer this message.")
    category: Category = Field(description="'allowed' when allowed, otherwise the block reason.")
    reason: str = Field(description="One short sentence explaining the decision.", max_length=300)


FAIL_CLOSED = Verdict(allowed=False, category="guardrails_unavailable", reason="no verdict")


class Guardrails(Protocol):
    async def classify(self, message: str, recent_questions: list[str]) -> Verdict: ...


def _escape(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


class AnthropicGuardrails:
    def __init__(self, spec: ModelSpec, system_prompt: str, client: AsyncAnthropic | None = None):
        self._spec = spec
        self._system = system_prompt
        self._client = client or AsyncAnthropic()

    async def classify(self, message: str, recent_questions: list[str]) -> Verdict:
        recent = "\n".join(f"- {_escape(q)}" for q in recent_questions[-3:]) or "(none)"
        content = (
            f"<recent_questions>\n{recent}\n</recent_questions>\n"
            f"<message>\n{_escape(message)}\n</message>"
        )
        response = await self._client.messages.parse(
            model=self._spec.model,
            max_tokens=self._spec.max_tokens,
            system=[{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"},
            output_config={"effort": self._spec.effort},
            output_format=Verdict,
        )
        verdict = response.parsed_output
        if response.stop_reason == "refusal" or verdict is None:
            return Verdict(allowed=False, category="harmful", reason="classifier declined")
        # A verdict that contradicts itself is treated as a block.
        if verdict.allowed != (verdict.category == "allowed"):
            log.warning("inconsistent verdict %s; failing closed", verdict)
            return Verdict(
                allowed=False,
                category=verdict.category if not verdict.allowed else "guardrails_unavailable",
                reason=verdict.reason,
            )
        return verdict
