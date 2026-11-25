import asyncio

import pytest

from gateway.agents.gate import Gate, GateBlocked, gated
from gateway.agents.guardrails import Verdict
from gateway.api.schemas import RunBlocked

ALLOW = Verdict(allowed=True, category="allowed", reason="ok")
DENY = Verdict(allowed=False, category="harmful", reason="no")


async def collect(agen):  # noqa: ANN001, ANN201
    return [x async for x in agen]


def classifier(verdict: Verdict, delay: float = 0.0, error: Exception | None = None):  # noqa: ANN201
    async def classify() -> Verdict:
        await asyncio.sleep(delay)
        if error:
            raise error
        return verdict

    return classify


async def test_events_buffered_until_allowed_then_released_in_order() -> None:
    seen_before_verdict = []

    async def agent():  # noqa: ANN202
        for i in range(5):
            yield i
            await asyncio.sleep(0)

    gate = Gate()
    out = []
    async for ev in gated(agent(), classifier(ALLOW, delay=0.05), gate, 1):
        seen_before_verdict.append(gate.released)
        out.append(ev)
    assert out == [0, 1, 2, 3, 4]
    assert all(seen_before_verdict)  # nothing reached the consumer before release


async def test_block_cancels_agent_and_emits_only_blocked() -> None:
    cancelled = asyncio.Event()

    async def agent():  # noqa: ANN202
        try:
            yield "secret partial answer"
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    out = await collect(gated(agent(), classifier(DENY, delay=0.02), Gate(), 1))
    assert len(out) == 1 and isinstance(out[0], RunBlocked)
    assert out[0].category == "harmful"
    assert cancelled.is_set()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delay": 5},  # timeout
        {"error": RuntimeError("classifier down")},  # classifier error
    ],
)
async def test_fail_closed(kwargs: dict) -> None:
    async def agent():  # noqa: ANN202
        yield "x"

    out = await collect(gated(agent(), classifier(ALLOW, **kwargs), Gate(), 0.1))
    assert [type(o) for o in out] == [RunBlocked]
    assert out[0].category == "guardrails_unavailable"


async def test_agent_error_surfaces_after_release() -> None:
    async def agent():  # noqa: ANN202
        yield 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await collect(gated(agent(), classifier(ALLOW), Gate(), 1))


async def test_wait_released() -> None:
    gate = Gate()
    waiter = asyncio.create_task(gate.wait_released())
    await asyncio.sleep(0.01)
    assert not waiter.done()  # held while undecided
    gate.decide(ALLOW)
    await asyncio.wait_for(waiter, 1)

    blocked = Gate()
    blocked.decide(DENY)
    with pytest.raises(GateBlocked):
        await blocked.wait_released()


async def test_guardrails_disabled_passthrough() -> None:
    async def agent():  # noqa: ANN202
        yield "a"

    gate = Gate()
    assert await collect(gated(agent(), None, gate, 1)) == ["a"]
    assert gate.released
