"""Turn queued events into Server-Sent Events frames, with a heartbeat comment so idle
proxies don't close the connection while the agent is thinking."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel


def frame(event: BaseModel) -> str:
    kind = getattr(event, "type", "message")
    return f"event: {kind}\ndata: {event.model_dump_json()}\n\n"


async def sse_stream(
    queue: asyncio.Queue[BaseModel | None], heartbeat_s: float
) -> AsyncIterator[str]:
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), heartbeat_s)
        except TimeoutError:
            yield ": heartbeat\n\n"
            continue
        if item is None:
            return
        yield frame(item)
