from __future__ import annotations

from fastapi import APIRouter, Request, Response

from gateway.api.schemas import ConversationOut, FeedbackRequest, TurnOut
from gateway.persistence.repository import Repository
from gateway.security.auth import RequiredPrincipal

router = APIRouter(prefix="/v1", tags=["history"])


def _repo(request: Request) -> Repository:
    return request.app.state.repo


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(principal: RequiredPrincipal, request: Request):
    return await _repo(request).list_conversations(principal.user_id)


@router.get("/conversations/{conversation_id}/messages", response_model=list[TurnOut])
async def list_messages(conversation_id: str, principal: RequiredPrincipal, request: Request):
    return await _repo(request).list_turns(principal.user_id, conversation_id)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, principal: RequiredPrincipal, request: Request):
    await _repo(request).delete_conversation(principal.user_id, conversation_id)
    return Response(status_code=204)


@router.post("/messages/{turn_id}/feedback", status_code=204)
async def add_feedback(
    turn_id: str, body: FeedbackRequest, principal: RequiredPrincipal, request: Request
):
    await _repo(request).add_feedback(principal.user_id, turn_id, body.rating, body.comment)
    return Response(status_code=204)
