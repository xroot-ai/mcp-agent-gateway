"""The gateway talks to MCP servers over real Streamable HTTP with bearer auth."""

import asyncio
import json
import socket
from pathlib import Path

import pytest
import uvicorn

from gateway.agents.gate import Gate
from gateway.agents.guardrails import Verdict
from gateway.config.loader import ConfigError, load_agent_config
from gateway.tools.context import RunContext
from gateway.tools.toolsets import ToolsetManager
from gateway.tools.wrapper import ToolExecutor
from mcp_servers.http_auth import BearerAuthMiddleware
from mcp_servers.retrieval.server import build_server
from tests.conftest import ROOT
from tests.fakes import FakeSearcher

TOKEN = "transport-test-token"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def serve(port: int, searcher: FakeSearcher) -> tuple[uvicorn.Server, asyncio.Task]:
    app = build_server(searcher=searcher).streamable_http_app(stateless_http=True, host="0.0.0.0")  # noqa: S104 - no rebinding check; bearer auth applies
    server = uvicorn.Server(
        uvicorn.Config(
            BearerAuthMiddleware(app, TOKEN), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    return server, task


def config(tmp_path: Path, port: int, token: str | None) -> Path:
    cfg = {
        "name": "http_test",
        "capabilities": [
            {
                "name": "docs",
                "url": f"http://127.0.0.1:{port}/mcp",
                "token": token,
                "tools": {"search": {"result_kind": "passages"}},
            }
        ],
        "prompts": {
            k: str(ROOT / f"configs/prompts/{k}.md")
            for k in ("orchestrator", "guardrails", "followups")
        },
    }
    p = tmp_path / "http.yaml"
    p.write_text(json.dumps(cfg))
    return p


async def test_search_over_http(tmp_path: Path) -> None:
    searcher, port = FakeSearcher(), free_port()
    server, task = await serve(port, searcher)
    manager = ToolsetManager()
    try:
        await manager.connect(load_agent_config(config(tmp_path, port, TOKEN)))
        gate = Gate()
        gate.decide(Verdict(allowed=True, category="allowed", reason=""))
        ex = ToolExecutor(manager, RunContext("r", "u", "c", "t"), gate)
        out = await ex.execute("search", {"query": "restart policy", "progress_title": "x"})
        assert not out.is_error and out.content.startswith("[1] Pod Lifecycle")
        assert searcher.queries[0][0] == "restart policy"
    finally:
        await manager.aclose()
        server.should_exit = True
        await task


async def test_startup_fails_with_wrong_token(tmp_path: Path) -> None:
    port = free_port()
    server, task = await serve(port, FakeSearcher())
    manager = ToolsetManager()
    try:
        with pytest.raises(Exception):  # noqa: B017 - transport-specific error type
            await manager.connect(load_agent_config(config(tmp_path, port, "wrong")))
    finally:
        await manager.aclose()
        server.should_exit = True
        await task


async def test_startup_fails_when_tool_missing(tmp_path: Path) -> None:
    port = free_port()
    server, task = await serve(port, FakeSearcher())
    cfg = json.loads(config(tmp_path, port, TOKEN).read_text())
    cfg["capabilities"][0]["tools"] = {"search_everything": {}}
    p = tmp_path / "missing.yaml"
    p.write_text(json.dumps(cfg))
    manager = ToolsetManager()
    try:
        with pytest.raises(ConfigError, match="search_everything"):
            await manager.connect(load_agent_config(p))
    finally:
        await manager.aclose()
        server.should_exit = True
        await task
