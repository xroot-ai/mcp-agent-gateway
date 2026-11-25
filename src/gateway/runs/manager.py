"""Each chat request becomes a run: a background task that writes events to a queue.

The HTTP response only reads the queue, so a client disconnect does not lose the turn
(it still completes and is saved), and ``DELETE /v1/chat/completions/{run_id}`` can
cancel the task from another request.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from gateway.agents.gate import Gate, gated
from gateway.agents.guardrails import Guardrails, Verdict
from gateway.agents.history_modes import build_history
from gateway.agents.orchestrator import Orchestrator, TurnInput
from gateway.api.errors import AppError
from gateway.api.schemas import (
    ChatRequest,
    RunBlocked,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    TextDelta,
)
from gateway.config.loader import AgentConfig
from gateway.observability import metrics
from gateway.observability.tracing import mark_interrupted, span
from gateway.persistence.models import new_id
from gateway.persistence.repository import Repository
from gateway.security.auth import Principal
from gateway.tools.context import RunContext
from gateway.tools.toolsets import ToolsetManager
from gateway.tools.wrapper import ToolExecutor

log = logging.getLogger(__name__)


class QuestionTooLong(AppError):
    status_code = 413
    code = "question_too_long"
    message = "The message is too long."


@dataclass
class RunHandle:
    run_id: str
    user_id: str
    conversation_id: str
    turn_id: str
    queue: asyncio.Queue[BaseModel | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None


class RunManager:
    def __init__(
        self,
        config: AgentConfig,
        repo: Repository,
        toolsets: ToolsetManager,
        orchestrator: Orchestrator,
        guardrails: Guardrails | None,
    ) -> None:
        self._config = config
        self._repo = repo
        self._toolsets = toolsets
        self._orchestrator = orchestrator
        self._guardrails = guardrails if config.features.guardrails else None
        self._runs: dict[str, RunHandle] = {}

    # ------------------------------------------------------------ public API
    async def start(self, principal: Principal, request: ChatRequest) -> RunHandle:
        question = request.message.strip()
        if len(question) > self._config.limits.max_question_chars:
            raise QuestionTooLong()
        conv = await self._repo.get_or_create_conversation(
            principal.user_id, request.conversation_id
        )
        run_id = new_id()
        turn = await self._repo.create_turn(conv.id, run_id, question)
        handle = RunHandle(run_id, principal.user_id, conv.id, turn.id)
        self._runs[run_id] = handle
        ctx = RunContext(run_id, principal.user_id, conv.id, turn.id, principal.scopes)
        handle.task = asyncio.create_task(
            self._execute(handle, ctx, question, want_title=conv.title is None),
            name=f"run-{run_id}",
        )
        return handle

    def cancel(self, run_id: str, user_id: str) -> bool:
        handle = self._runs.get(run_id)
        if handle is None or handle.user_id != user_id or handle.task is None:
            return False  # unknown and not-yours look the same to the caller
        handle.task.cancel()
        return True

    async def shutdown(self) -> None:
        tasks = [h.task for h in self._runs.values() if h.task and not h.task.done()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------ the run itself
    async def _execute(
        self, h: RunHandle, ctx: RunContext, question: str, want_title: bool
    ) -> None:
        put = h.queue.put_nowait
        partial: list[str] = []
        status = "failed"
        with span("run", run_id=h.run_id, conversation_id=h.conversation_id) as s:
            try:
                put(
                    RunStarted(
                        run_id=h.run_id, conversation_id=h.conversation_id, turn_id=h.turn_id
                    )
                )
                async with asyncio.timeout(self._config.limits.run_timeout_s):
                    turn_input = await self._turn_input(h.conversation_id, question, want_title)
                    gate = Gate()
                    executor = ToolExecutor(self._toolsets, ctx, gate)
                    classify = self._classifier(question, turn_input)
                    events = self._orchestrator.run(turn_input, executor)
                    async for ev in gated(
                        events, classify, gate, self._config.limits.guardrails_timeout_s
                    ):
                        if isinstance(ev, TextDelta):
                            partial.append(ev.text)
                        elif isinstance(ev, RunCompleted):
                            await self._repo.finish_turn(
                                h.turn_id,
                                status="completed",
                                answer=ev.answer,
                                sources=ev.sources,
                                followups=ev.followups,
                            )
                            if ev.title:
                                await self._repo.set_title_if_missing(h.conversation_id, ev.title)
                            status = "completed"
                        elif isinstance(ev, RunBlocked):
                            await self._repo.finish_turn(
                                h.turn_id, status="blocked", block_category=ev.category
                            )
                            mark_interrupted(s, f"blocked:{ev.category}")
                            status = "blocked"
                        put(ev)
                    if status == "failed":  # stream ended without a terminal event
                        raise RuntimeError("agent finished without a result")
            except TimeoutError:
                status = "timeout"
                mark_interrupted(s, "timeout")
                await self._finish_quietly(h, "failed", "".join(partial))
                put(RunFailed(code="timeout", message="The request took too long."))
            except asyncio.CancelledError:
                status = "cancelled"
                mark_interrupted(s, "cancelled")
                await self._finish_quietly(h, "cancelled", "".join(partial))
                put(RunCancelled())
            except Exception:
                log.exception("run %s failed", h.run_id)
                status = "failed"
                await self._finish_quietly(h, "failed", "".join(partial))
                put(RunFailed(code="internal_error", message="Something went wrong."))
            finally:
                s.set_attribute("run.status", status)
                metrics.runs.add(1, {"status": status})
                put(None)
                self._runs.pop(h.run_id, None)

    def _classifier(
        self, question: str, turn_input: TurnInput
    ) -> Callable[[], Awaitable[Verdict]] | None:
        guardrails = self._guardrails
        if guardrails is None:
            return None
        recent = [str(m["content"]) for m in turn_input.history if m["role"] == "user"]

        def classify() -> Awaitable[Verdict]:
            return guardrails.classify(question, recent)

        return classify

    async def _finish_quietly(self, h: RunHandle, status: str, answer: str) -> None:
        with contextlib.suppress(Exception):
            await asyncio.shield(self._repo.finish_turn(h.turn_id, status=status, answer=answer))

    async def _turn_input(self, conversation_id: str, question: str, want_title: bool) -> TurnInput:
        hc = self._config.history
        prior = await self._repo.prior_turns(conversation_id, limit=max(hc.max_turns, 1) * 2)
        previous_sources = []
        if hc.include_previous_sources and hc.mode != "none":
            previous_sources = await self._repo.last_sources(conversation_id)
        return TurnInput(
            question=question,
            history=build_history(prior, hc.mode, hc.max_turns),
            previous_sources=previous_sources,
            want_title=want_title,
        )
