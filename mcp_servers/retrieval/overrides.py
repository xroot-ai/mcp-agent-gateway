"""Per-call search overrides: an allowlist, each value validated and bounded.

Anything not on the list (a collection name, a backend URL, an API key, a raw filter)
is rejected. The model can narrow a search; it cannot redirect one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class OverrideError(ValueError):
    pass


_SECTION = re.compile(r"^[a-z0-9][a-z0-9/_-]{0,79}$")


@dataclass(frozen=True)
class SearchOverrides:
    section_prefix: str | None = None
    min_score: float | None = None


def _section_prefix(value: Any) -> str:
    if not isinstance(value, str) or not _SECTION.fullmatch(value) or ".." in value:
        raise OverrideError("section_prefix must look like 'concepts/workloads'")
    return value


def _min_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise OverrideError("min_score must be a number between 0 and 1")
    return float(value)


ALLOWED = {"section_prefix": _section_prefix, "min_score": _min_score}


def parse_overrides(raw: dict[str, Any] | None) -> SearchOverrides:
    if not raw:
        return SearchOverrides()
    if not isinstance(raw, dict):
        raise OverrideError("overrides must be an object")
    unknown = sorted(set(raw) - set(ALLOWED))
    if unknown:
        raise OverrideError(f"override(s) not allowed: {unknown}; allowed: {sorted(ALLOWED)}")
    return SearchOverrides(**{k: ALLOWED[k](v) for k, v in raw.items()})
