"""Typed source contract between retrieval tools and the answer.

Retrieval tools return ``{"passages": [{"title", "url", "section", "text", "score"}]}``.
The registry de-duplicates passages across tool calls (same url + section is one source),
assigns stable numbers in first-seen order, and renders what the model reads with those
numbers so it can cite ``[n]``. The final event carries only sources the answer cites;
if it cites none, all retrieved sources are returned so the client can still show them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from gateway.api.schemas import Source


class Passage(BaseModel):
    title: str = Field(max_length=300)
    url: str = Field(max_length=500)
    section: str = Field(default="", max_length=300)
    text: str
    score: float | None = None


class PassageResult(BaseModel):
    passages: list[Passage]


_CITATION = re.compile(r"\[(\d{1,3})\]")


def parse_passages(payload: Any) -> list[Passage]:
    """Validate a tool result against the passage contract; raises ValueError if it does not fit."""
    try:
        return PassageResult.model_validate(payload).passages
    except ValidationError as exc:
        raise ValueError("retrieval tool returned data outside the passage contract") from exc


@dataclass
class SourceRegistry:
    _by_key: dict[tuple[str, str], Source] = field(default_factory=dict)

    def add(self, passages: list[Passage]) -> list[tuple[Source, Passage]]:
        numbered = []
        for p in passages:
            key = (p.url.strip().rstrip("/").lower(), p.section.strip().lower())
            src = self._by_key.get(key)
            if src is None:
                src = Source(
                    number=len(self._by_key) + 1,
                    title=p.title,
                    url=p.url,
                    section=p.section,
                    snippet=p.text[:240],
                )
                self._by_key[key] = src
            numbered.append((src, p))
        return numbered

    def render_for_model(self, numbered: list[tuple[Source, Passage]]) -> str:
        if not numbered:
            return "No matching passages were found."
        blocks = []
        for src, p in numbered:
            heading = f"[{src.number}] {p.title}" + (f" > {p.section}" if p.section else "")
            blocks.append(f"{heading}\n{p.text}")
        return "\n\n".join(blocks)

    @property
    def all(self) -> list[Source]:
        return sorted(self._by_key.values(), key=lambda s: s.number)

    def cited_in(self, answer: str) -> list[Source]:
        cited = {int(n) for n in _CITATION.findall(answer)}
        chosen = [s for s in self.all if s.number in cited]
        return chosen or self.all


def previous_sources_context(sources: list[Source], limit: int = 8) -> str:
    """Context block for follow-up turns: what the previous answer cited, so "the second
    link" or "that page" can be resolved. Numbers here are from the previous answer only."""
    if not sources:
        return ""
    lines = [
        f"- (prev {s.number}) {s.title}" + (f" > {s.section}" if s.section else "") + f" <{s.url}>"
        for s in sources[:limit]
    ]
    return "Sources cited in the previous answer:\n" + "\n".join(lines)
