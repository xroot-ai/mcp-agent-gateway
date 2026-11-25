"""Request/response models and the typed SSE event contract."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- sources
class Source(BaseModel):
    """A retrieved passage the answer may cite as ``[number]``."""

    number: int
    title: str
    url: str
    section: str = ""
    snippet: str = ""


# ---------------------------------------------------------------- SSE events
class RunStarted(BaseModel):
    type: Literal["run.started"] = "run.started"
    run_id: str
    conversation_id: str
    turn_id: str


class Progress(BaseModel):
    type: Literal["progress"] = "progress"
    tool: str
    title: str
    detail: str = ""


class TextDelta(BaseModel):
    type: Literal["text.delta"] = "text.delta"
    text: str


class TitleEvent(BaseModel):
    type: Literal["title"] = "title"
    title: str


class RunCompleted(BaseModel):
    type: Literal["run.completed"] = "run.completed"
    answer: str
    sources: list[Source] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    title: str | None = None


class RunBlocked(BaseModel):
    type: Literal["run.blocked"] = "run.blocked"
    category: str
    message: str


class RunCancelled(BaseModel):
    type: Literal["run.cancelled"] = "run.cancelled"


class RunFailed(BaseModel):
    type: Literal["run.failed"] = "run.failed"
    code: str
    message: str


Event = Annotated[
    RunStarted
    | Progress
    | TextDelta
    | TitleEvent
    | RunCompleted
    | RunBlocked
    | RunCancelled
    | RunFailed,
    Field(discriminator="type"),
]

TERMINAL_EVENTS = (RunCompleted, RunBlocked, RunCancelled, RunFailed)


# ---------------------------------------------------------------- requests
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = Field(default=None, max_length=64)


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------- responses
class ConversationOut(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class TurnOut(BaseModel):
    id: str
    question: str
    answer: str
    status: str
    sources: list[Source]
    followups: list[str]
    created_at: datetime
