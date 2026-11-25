"""Conversation storage: Conversation 1-n Turn 1-n Source, Turn 1-n Feedback."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class Conversation(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True, max_length=64)
    user_id: str = Field(index=True, max_length=128)
    title: str | None = Field(default=None, max_length=200)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Turn(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True, max_length=64)
    conversation_id: str = Field(foreign_key="conversation.id", index=True, ondelete="CASCADE")
    run_id: str = Field(index=True, max_length=64)
    question: str
    answer: str = ""
    # running | completed | blocked | cancelled | failed
    status: str = Field(default="running", max_length=16)
    block_category: str | None = Field(default=None, max_length=32)
    followups: str = "[]"  # JSON list
    created_at: datetime = Field(default_factory=_now)


class SourceRow(SQLModel, table=True):
    __tablename__ = "source"  # type: ignore[assignment]

    id: int | None = Field(default=None, primary_key=True)
    turn_id: str = Field(foreign_key="turn.id", index=True, ondelete="CASCADE")
    number: int
    title: str
    url: str
    section: str = ""
    snippet: str = ""


class Feedback(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    turn_id: str = Field(foreign_key="turn.id", index=True, ondelete="CASCADE")
    user_id: str = Field(max_length=128)
    rating: str = Field(max_length=8)
    comment: str | None = None
    created_at: datetime = Field(default_factory=_now)
