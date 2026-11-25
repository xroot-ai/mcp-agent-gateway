"""Per-call context: values the gateway supplies to tools, never the model.

A tool policy maps argument names to context keys, e.g. ``{"requester": "user_id"}``.
The argument is removed from the schema the model sees, the model may not supply it,
and a missing value raises instead of silently sending ``None`` downstream.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


class ToolContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunContext:
    run_id: str
    user_id: str
    conversation_id: str
    turn_id: str
    scopes: frozenset[str] = field(default_factory=frozenset)

    def lookup(self, key: str) -> str:
        value = getattr(self, key, None) if key in _ALLOWED_KEYS else None
        if not isinstance(value, str) or not value:
            raise ToolContextError(f"context value {key!r} is not available for this run")
        return value


_ALLOWED_KEYS = frozenset({"run_id", "user_id", "conversation_id", "turn_id"})


def resolve_context(mapping: dict[str, str], ctx: RunContext) -> dict[str, str]:
    return {arg: ctx.lookup(key) for arg, key in mapping.items()}


def hide_context_args(schema: dict[str, Any], context_args: set[str]) -> dict[str, Any]:
    """Return a copy of a JSON schema with gateway-supplied arguments removed."""
    if not context_args:
        return copy.deepcopy(schema)
    out = copy.deepcopy(schema)
    props = out.get("properties", {})
    missing = context_args - set(props)
    if missing:
        raise ToolContextError(f"tool schema has no argument(s) {sorted(missing)} to fill")
    for name in context_args:
        props.pop(name, None)
    if "required" in out:
        out["required"] = [r for r in out["required"] if r not in context_args]
    return out
