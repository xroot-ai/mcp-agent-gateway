from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from gateway.api.errors import NotFound
from gateway.api.schemas import ChatRequest
from gateway.runs.manager import RunManager
from gateway.runs.sse import sse_stream
from gateway.security.auth import RequiredPrincipal

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_class=StreamingResponse)
async def create_completion(body: ChatRequest, principal: RequiredPrincipal, request: Request):
    """Start a run and stream its events as SSE.

    Events: run.started, progress, title, text.delta, then exactly one of run.completed,
    run.blocked, run.cancelled, run.failed.
    """
    runs: RunManager = request.app.state.runs
    handle = await runs.start(principal, body)
    return StreamingResponse(
        sse_stream(handle.queue, request.app.state.settings.sse_heartbeat_s),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Run-Id": handle.run_id},
    )


@router.delete("/chat/completions/{run_id}", status_code=202)
async def cancel_completion(run_id: str, principal: RequiredPrincipal, request: Request):
    """Cancel a running run you own. The stream ends with a run.cancelled event."""
    runs: RunManager = request.app.state.runs
    if not runs.cancel(run_id, principal.user_id):
        raise NotFound("Run not found.")
    return {"run_id": run_id, "status": "cancelling"}
