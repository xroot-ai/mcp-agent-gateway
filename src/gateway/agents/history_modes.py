"""How much prior conversation the orchestrator sees.

- ``questions_only``: earlier user questions, no earlier answers. Cheap, and keeps stale
  answers from being repeated as fact; follow-ups like "and for StatefulSets?" still resolve.
- ``first_and_last``: the first exchange (sets the topic) and the last exchange (immediate
  context), with answers.
- ``none``: every question stands alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

HistoryMode = Literal["questions_only", "first_and_last", "none"]


@dataclass(frozen=True)
class PriorTurn:
    question: str
    answer: str


def build_history(
    turns: list[PriorTurn], mode: HistoryMode, max_turns: int
) -> list[dict[str, Any]]:
    """Return Messages API messages to prepend before the new question."""
    turns = [t for t in turns if t.answer.strip()]  # skip blocked/cancelled/failed turns
    if mode == "none" or max_turns == 0 or not turns:
        return []
    if mode == "questions_only":
        recent = turns[-max_turns:]
        listed = "\n".join(f"{i}. {t.question}" for i, t in enumerate(recent, 1))
        return [
            {"role": "user", "content": f"<earlier_questions>\n{listed}\n</earlier_questions>"},
            {"role": "assistant", "content": "Noted. I'll use these for context."},
        ]
    if mode == "first_and_last":
        chosen = [turns[0]] if len(turns) == 1 else [turns[0], turns[-1]]
        messages: list[dict[str, Any]] = []
        for t in chosen:
            messages.append({"role": "user", "content": t.question})
            messages.append({"role": "assistant", "content": t.answer})
        return messages
    raise ValueError(f"unknown history mode {mode!r}")
