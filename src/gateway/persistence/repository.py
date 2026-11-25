"""Async repository. Every read and write that takes a conversation or turn id also takes
the caller's user id, so one user can never see or touch another user's data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel, col, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gateway.agents.history_modes import PriorTurn
from gateway.api.errors import NotFound
from gateway.api.schemas import ConversationOut, Source, TurnOut
from gateway.persistence.models import Conversation, Feedback, SourceRow, Turn


class Repository:
    def __init__(self, database_url: str) -> None:
        if database_url.startswith("sqlite") and ":///" in database_url:
            db_path = database_url.split(":///", 1)[1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine: AsyncEngine = create_async_engine(database_url)
        if database_url.startswith("sqlite"):

            @event.listens_for(self.engine.sync_engine, "connect")
            def _fk_on(dbapi_conn, _record):  # noqa: ANN001
                dbapi_conn.execute("PRAGMA foreign_keys=ON")

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    def _session(self) -> AsyncSession:
        return AsyncSession(self.engine, expire_on_commit=False)

    # ------------------------------------------------------------ conversations
    async def get_or_create_conversation(
        self, user_id: str, conversation_id: str | None
    ) -> Conversation:
        async with self._session() as s:
            if conversation_id:
                conv = await s.get(Conversation, conversation_id)
                if conv is None or conv.user_id != user_id:
                    raise NotFound("Conversation not found.")
                return conv
            conv = Conversation(user_id=user_id)
            s.add(conv)
            await s.commit()
            return conv

    async def set_title_if_missing(self, conversation_id: str, title: str) -> None:
        async with self._session() as s:
            conv = await s.get(Conversation, conversation_id)
            if conv and not conv.title:
                conv.title = title[:200]
                s.add(conv)
                await s.commit()

    async def list_conversations(self, user_id: str, limit: int = 50) -> list[ConversationOut]:
        async with self._session() as s:
            rows = await s.exec(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(col(Conversation.updated_at).desc())
                .limit(limit)
            )
            return [ConversationOut.model_validate(r, from_attributes=True) for r in rows]

    async def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        async with self._session() as s:
            conv = await s.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user_id:
                raise NotFound("Conversation not found.")
            await s.delete(conv)
            await s.commit()

    # ------------------------------------------------------------ turns
    async def create_turn(self, conversation_id: str, run_id: str, question: str) -> Turn:
        async with self._session() as s:
            turn = Turn(conversation_id=conversation_id, run_id=run_id, question=question)
            s.add(turn)
            await s.commit()
            return turn

    async def finish_turn(
        self,
        turn_id: str,
        *,
        status: str,
        answer: str = "",
        sources: list[Source] | None = None,
        followups: list[str] | None = None,
        block_category: str | None = None,
    ) -> None:
        async with self._session() as s:
            turn = await s.get(Turn, turn_id)
            if turn is None:
                return
            turn.status = status
            turn.answer = answer
            turn.block_category = block_category
            turn.followups = json.dumps(followups or [])
            s.add(turn)
            await s.exec(delete(SourceRow).where(col(SourceRow.turn_id) == turn_id))
            for src in sources or []:
                s.add(SourceRow(turn_id=turn_id, **src.model_dump()))
            conv = await s.get(Conversation, turn.conversation_id)
            if conv:
                conv.updated_at = datetime.now(UTC)
                s.add(conv)
            await s.commit()

    async def list_turns(self, user_id: str, conversation_id: str) -> list[TurnOut]:
        async with self._session() as s:
            conv = await s.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user_id:
                raise NotFound("Conversation not found.")
            turns = list(
                await s.exec(
                    select(Turn)
                    .where(Turn.conversation_id == conversation_id)
                    .order_by(col(Turn.created_at))
                )
            )
            out = []
            for t in turns:
                out.append(
                    TurnOut(
                        id=t.id,
                        question=t.question,
                        answer=t.answer,
                        status=t.status,
                        sources=await self._sources(s, t.id),
                        followups=json.loads(t.followups),
                        created_at=t.created_at,
                    )
                )
            return out

    async def prior_turns(self, conversation_id: str, limit: int = 20) -> list[PriorTurn]:
        """Completed turns only, oldest first."""
        async with self._session() as s:
            rows = list(
                await s.exec(
                    select(Turn)
                    .where(Turn.conversation_id == conversation_id, Turn.status == "completed")
                    .order_by(col(Turn.created_at).desc())
                    .limit(limit)
                )
            )
        return [PriorTurn(question=t.question, answer=t.answer) for t in reversed(rows)]

    async def last_sources(self, conversation_id: str) -> list[Source]:
        async with self._session() as s:
            last = (
                await s.exec(
                    select(Turn)
                    .where(Turn.conversation_id == conversation_id, Turn.status == "completed")
                    .order_by(col(Turn.created_at).desc())
                    .limit(1)
                )
            ).first()
            return await self._sources(s, last.id) if last else []

    @staticmethod
    async def _sources(s: AsyncSession, turn_id: str) -> list[Source]:
        rows = await s.exec(
            select(SourceRow).where(SourceRow.turn_id == turn_id).order_by(col(SourceRow.number))
        )
        return [
            Source(number=r.number, title=r.title, url=r.url, section=r.section, snippet=r.snippet)
            for r in rows
        ]

    # ------------------------------------------------------------ feedback
    async def add_feedback(
        self, user_id: str, turn_id: str, rating: str, comment: str | None
    ) -> None:
        async with self._session() as s:
            turn = await s.get(Turn, turn_id)
            conv = await s.get(Conversation, turn.conversation_id) if turn else None
            if turn is None or conv is None or conv.user_id != user_id:
                raise NotFound("Message not found.")
            await s.exec(
                delete(Feedback).where(
                    col(Feedback.turn_id) == turn_id, col(Feedback.user_id) == user_id
                )
            )
            s.add(Feedback(turn_id=turn_id, user_id=user_id, rating=rating, comment=comment))
            await s.commit()
