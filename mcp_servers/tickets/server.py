"""Tickets MCP server: ``create_ticket``, a side-effecting demo tool backed by SQLite.

The gateway marks this tool ``side_effect: true`` and fills ``requester`` and
``conversation_ref`` itself, so the model can neither impersonate another user nor
file a ticket before guardrails have approved the request.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.requests import Request
from starlette.responses import JSONResponse

log = logging.getLogger("tickets")


class TicketSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TICKETS_", env_file=".env", extra="ignore")

    db_path: str = "data/tickets.db"
    token: str = ""
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8102


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL,
    priority TEXT NOT NULL, requester TEXT NOT NULL, conversation_ref TEXT NOT NULL,
    created_at TEXT NOT NULL
)"""


def _insert(db_path: str, row: tuple[str, ...]) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)
        conn.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?)", row)


def list_tickets(db_path: str) -> list[dict[str, str]]:
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(_SCHEMA)
        return [dict(r) for r in conn.execute("SELECT * FROM tickets ORDER BY created_at")]


def build_server(settings: TicketSettings | None = None) -> MCPServer:
    settings = settings or TicketSettings()
    server = MCPServer("tickets", instructions="File support tickets.")

    @server.tool(
        description=(
            "Open a support ticket for the user. Only call this when the user explicitly asks "
            "for a ticket, and after confirming the title and description with them."
        )
    )
    async def create_ticket(
        title: Annotated[str, Field(min_length=5, max_length=120)],
        description: Annotated[str, Field(min_length=10, max_length=4000)],
        requester: Annotated[str, Field(min_length=1, max_length=128)],
        conversation_ref: Annotated[str, Field(min_length=1, max_length=64)],
        priority: Literal["low", "normal", "high"] = "normal",
    ) -> dict[str, str]:
        ticket_id = "TCK-" + uuid.uuid4().hex[:8].upper()
        row = (
            ticket_id,
            title,
            description,
            priority,
            requester,
            conversation_ref,
            datetime.now(UTC).isoformat(),
        )
        await asyncio.to_thread(_insert, settings.db_path, row)
        log.info("ticket %s created for %s", ticket_id, requester)
        return {"ticket_id": ticket_id, "status": "open"}

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return server


def http_app(settings: TicketSettings | None = None):  # noqa: ANN201
    from mcp_servers.http_auth import BearerAuthMiddleware

    settings = settings or TicketSettings()
    app = build_server(settings).streamable_http_app(stateless_http=True, host=settings.host)
    return BearerAuthMiddleware(app, settings.token, open_paths=frozenset({"/healthz"}))


if __name__ == "__main__":
    import uvicorn

    s = TicketSettings()
    logging.basicConfig(level="INFO")
    uvicorn.run(http_app(s), host=s.host, port=s.port)
