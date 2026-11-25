from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.config.settings import Settings
from gateway.main import create_app
from gateway.tools.toolsets import register_local_server
from mcp_servers.retrieval.server import build_server as build_retrieval
from mcp_servers.tickets.server import TicketSettings
from mcp_servers.tickets.server import build_server as build_tickets
from tests.fakes import FakeGuard, FakeLLM, FakeSearcher

ROOT = Path(__file__).resolve().parents[1]
ALICE = "alice-token"
BOB = "bob-token"
TOKENS = {
    ALICE: {"user_id": "alice", "scopes": ["tickets:write"]},
    BOB: {"user_id": "bob", "scopes": []},
}


def write_config(tmp_path: Path, **overrides: object) -> Path:
    prompts = ROOT / "configs" / "prompts"
    cfg = {
        "name": "test_agent",
        "limits": {"max_tool_rounds": 3, "run_timeout_s": 10, "guardrails_timeout_s": 2},
        "history": {"mode": "questions_only", "max_turns": 4},
        "capabilities": [
            {
                "name": "docs",
                "url": "local://test_docs",
                "tools": {"search": {"max_calls": 3, "progress": True, "result_kind": "passages"}},
            },
            {
                "name": "tickets",
                "url": "local://test_tickets",
                "tools": {
                    "create_ticket": {
                        "side_effect": True,
                        "required_scope": "tickets:write",
                        "max_calls": 1,
                        "context": {"requester": "user_id", "conversation_ref": "conversation_id"},
                    }
                },
            },
        ],
        "prompts": {
            k: str(prompts / f"{k}.md") for k in ("orchestrator", "guardrails", "followups")
        },
    }
    cfg.update(overrides)
    path = tmp_path / "agent.yaml"
    path.write_text(json.dumps(cfg))  # JSON is valid YAML
    return path


@pytest.fixture
def searcher() -> FakeSearcher:
    return FakeSearcher()


@pytest.fixture
def tickets_db(tmp_path: Path) -> str:
    return str(tmp_path / "tickets.db")


@pytest.fixture(autouse=True)
def local_servers(searcher: FakeSearcher, tickets_db: str) -> None:
    register_local_server("test_docs", lambda: build_retrieval(searcher=searcher))
    register_local_server("test_tickets", lambda: build_tickets(TicketSettings(db_path=tickets_db)))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        agent_config=str(write_config(tmp_path)),
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gw.db'}",
        api_tokens=TOKENS,
        sse_heartbeat_s=5,
    )


def make_client(settings: Settings, llm: FakeLLM, guard: FakeGuard | None = None) -> TestClient:
    return TestClient(create_app(settings, llm=llm, guardrails=guard or FakeGuard()))


@pytest.fixture
def client_factory(settings: Settings) -> Iterator:
    clients: list[TestClient] = []

    def factory(llm: FakeLLM, guard: FakeGuard | None = None) -> TestClient:
        c = make_client(settings, llm, guard)
        c.__enter__()
        clients.append(c)
        return c

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


def auth(token: str = ALICE) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def read_events(response) -> list[dict]:  # noqa: ANN001
    events = []
    for block in response.text.split("\n\n"):
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads("".join(data)))
    return events
