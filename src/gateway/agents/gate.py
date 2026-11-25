"""Run the agent and the guardrails check in parallel; release nothing until the verdict.

- Agent events are buffered until the verdict arrives, then flushed in order.
- Blocked: the agent task is cancelled and the client sees only a ``run.blocked`` event.
- Timeout or classifier error: treated as blocked (fail closed).
- Side-effecting tools call ``Gate.wait_released()`` before they run, so a blocked or
  undecided request can read docs but can never file a ticket.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from gateway.agents.guardrails import BLOCK_MESSAGES, FAIL_CLOSED, Verdict
from gateway.api.schemas import RunBlocked

log = logging.getLogger(__name__)


class GateBlocked(Exception):
    pass


class Gate:
    def __init__(self) -> None:
        self._decided = asyncio.Event()
        self.verdict: Verdict | None = None

    @property
    def released(self) -> bool:
        return self._decided.is_set() and bool(self.verdict and self.verdict.allowed)

    def decide(self, verdict: Verdict) -> None:
        self.verdict = verdict
        self._decided.set()

    async def wait_released(self) -> None:
        await self._decided.wait()
        if not (self.verdict and self.verdict.allowed):
            raise GateBlocked("request was blocked by guardrails")


class _Done:
    pass


class _Failure:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


async def _verdict(classify: Callable[[], Awaitable[Verdict]], timeout_s: float) -> Verdict:
    try:
        return await asyncio.wait_for(classify(), timeout_s)
    except TimeoutError:
        log.warning("guardrails timed out after %.1fs; failing closed", timeout_s)
    except Exception:
        log.exception("guardrails failed; failing closed")
    return FAIL_CLOSED


def blocked_event(verdict: Verdict) -> RunBlocked:
    category = verdict.category if verdict.category != "allowed" else "guardrails_unavailable"
    return RunBlocked(
        category=category, message=BLOCK_MESSAGES.get(category, BLOCK_MESSAGES["harmful"])
    )


async def gated[T](
    events: AsyncIterator[T],
    classify: Callable[[], Awaitable[Verdict]] | None,
    gate: Gate,
    timeout_s: float,
) -> AsyncIterator[T | RunBlocked]:
    if classify is None:  # guardrails disabled in config
        gate.decide(Verdict(allowed=True, category="allowed", reason="guardrails disabled"))
        async for event in events:
            yield event
        return

    queue: asyncio.Queue[T | _Done | _Failure] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for event in events:
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # forwarded to the consumer once released
            await queue.put(_Failure(exc))
        else:
            await queue.put(_Done())

    producer = asyncio.create_task(pump(), name="agent")
    guard = asyncio.create_task(_verdict(classify, timeout_s), name="guardrails")
    buffered: list[T | _Done | _Failure] = []
    try:
        while not guard.done():
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({guard, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                buffered.append(getter.result())
            else:
                getter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await getter
        verdict = guard.result()
        gate.decide(verdict)
        if not verdict.allowed:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            yield blocked_event(verdict)
            return
        pending = list(buffered)
        while True:
            item = pending.pop(0) if pending else await queue.get()
            if isinstance(item, _Done):
                return
            if isinstance(item, _Failure):
                raise item.exc
            yield item
    finally:
        for task in (producer, guard):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
