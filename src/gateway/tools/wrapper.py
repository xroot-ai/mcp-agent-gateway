"""Every tool call goes through here: permissions, per-tool limits, argument validation,
context injection, side-effect hold, timeout, metrics and result shaping."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaError

from gateway.agents.gate import Gate, GateBlocked
from gateway.agents.sources import SourceRegistry, parse_passages
from gateway.observability import metrics
from gateway.observability.tracing import span
from gateway.tools.context import RunContext, ToolContextError, resolve_context
from gateway.tools.toolsets import ToolsetManager

log = logging.getLogger(__name__)


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False


@dataclass
class ToolExecutor:
    toolsets: ToolsetManager
    ctx: RunContext
    gate: Gate
    sources: SourceRegistry = field(default_factory=SourceRegistry)
    calls: Counter[str] = field(default_factory=Counter)

    async def execute(self, name: str, args: Any) -> ToolOutcome:
        started = time.perf_counter()
        outcome = "error"
        with span("tool.call", tool=name, run_id=self.ctx.run_id) as s:
            try:
                result = await self._execute(name, args)
                outcome = "error" if result.is_error else "ok"
                return result
            except GateBlocked:
                outcome = "held_blocked"
                return ToolOutcome("This action is not permitted for this request.", True)
            finally:
                s.set_attribute("tool.outcome", outcome)
                metrics.tool_calls.add(1, {"tool": name, "outcome": outcome})
                metrics.tool_latency.record(time.perf_counter() - started, {"tool": name})

    async def _execute(self, name: str, args: Any) -> ToolOutcome:
        binding = self.toolsets.bindings.get(name)
        if binding is None:
            return ToolOutcome(f"Unknown tool {name!r}.", True)
        policy = binding.policy

        # 1. permission
        if policy.required_scope and policy.required_scope not in self.ctx.scopes:
            return ToolOutcome("You do not have permission to use this tool.", True)

        # 2. per-run limit (counted before validation so retries with bad args still count)
        self.calls[name] += 1
        if self.calls[name] > policy.max_calls:
            return ToolOutcome(
                f"Call limit reached for {name} ({policy.max_calls} per request). "
                "Answer with what you already have.",
                True,
            )

        # 3. arguments: must be an object, must not set gateway-owned args, must match schema
        if not isinstance(args, dict):
            return ToolOutcome("Tool input must be a JSON object.", True)
        owned = sorted(set(args) & set(policy.context))
        if owned:
            return ToolOutcome(f"Argument(s) {owned} are set by the gateway, not the model.", True)
        unknown = sorted(set(args) - set(binding.model_schema.get("properties", {})))
        if unknown:
            return ToolOutcome(f"Unknown argument(s) {unknown}.", True)
        try:
            Draft202012Validator(binding.model_schema).validate(args)
        except SchemaError as exc:
            return ToolOutcome(f"Invalid arguments: {exc.message}", True)

        # 4. context injection (raises loudly if a value is missing)
        try:
            call_args = {**args, **resolve_context(policy.context, self.ctx)}
        except ToolContextError:
            log.exception("missing tool context for %s", name)
            return ToolOutcome("This tool is unavailable for this request.", True)

        # 5. side-effect hold: wait for the guardrails verdict before acting
        if policy.side_effect:
            await self.gate.wait_released()

        # 6. call with timeout
        try:
            result = await asyncio.wait_for(
                self.toolsets.call(binding, call_args), policy.timeout_s
            )
        except TimeoutError:
            return ToolOutcome(f"{name} timed out.", True)
        except Exception:
            log.exception("tool %s failed", name)
            return ToolOutcome(f"{name} failed.", True)
        if result.is_error:
            return ToolOutcome(result.text or f"{name} failed.", True)

        # 7. shape the result
        if policy.result_kind == "passages":
            try:
                passages = parse_passages(result.data)
            except ValueError:
                log.warning("tool %s broke the passage contract", name)
                return ToolOutcome(f"{name} returned an unreadable result.", True)
            return ToolOutcome(self.sources.render_for_model(self.sources.add(passages)))
        return ToolOutcome(result.text)
