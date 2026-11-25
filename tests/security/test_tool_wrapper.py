"""Rules enforced on every tool call by gateway.tools.wrapper."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from gateway.agents.gate import Gate
from gateway.agents.guardrails import Verdict
from gateway.config.loader import load_agent_config
from gateway.tools.context import RunContext
from gateway.tools.toolsets import ToolsetManager
from gateway.tools.wrapper import ToolExecutor
from mcp_servers.tickets.server import list_tickets
from tests.conftest import write_config

TICKET = {"title": "Pods stuck pending", "description": "Pods in ns demo never schedule."}
ALLOW = Verdict(allowed=True, category="allowed", reason="ok")
DENY = Verdict(allowed=False, category="harmful", reason="no")


@asynccontextmanager
async def connected(tmp_path: Path) -> AsyncIterator[ToolsetManager]:
    # Open and close in the same task: MCP clients use anyio cancel scopes.
    manager = ToolsetManager()
    try:
        await manager.connect(load_agent_config(write_config(tmp_path)))
        yield manager
    finally:
        await manager.aclose()


def executor(
    toolsets: ToolsetManager, scopes=frozenset({"tickets:write"}), gate=None
) -> ToolExecutor:  # noqa: ANN001
    ctx = RunContext("run1", "alice", "conv1", "turn1", scopes)
    g = gate or Gate()
    if gate is None:
        g.decide(ALLOW)
    return ToolExecutor(toolsets, ctx, g)


async def test_context_args_hidden_from_model(tmp_path: Path) -> None:
    async with connected(tmp_path) as toolsets:
        schema = toolsets.bindings["create_ticket"].model_schema
        assert "requester" not in schema["properties"]
        assert "conversation_ref" not in schema["properties"]


async def test_model_cannot_set_context_args(tmp_path: Path, tickets_db: str) -> None:
    async with connected(tmp_path) as toolsets:
        out = await executor(toolsets).execute("create_ticket", {**TICKET, "requester": "bob"})
        assert out.is_error and "set by the gateway" in out.content
        assert list_tickets(tickets_db) == []


async def test_context_injected_from_run(tmp_path: Path, tickets_db: str) -> None:
    async with connected(tmp_path) as toolsets:
        out = await executor(toolsets).execute("create_ticket", TICKET)
        assert not out.is_error, out.content
        (row,) = list_tickets(tickets_db)
        assert row["requester"] == "alice" and row["conversation_ref"] == "conv1"


async def test_scope_required(tmp_path: Path, tickets_db: str) -> None:
    async with connected(tmp_path) as toolsets:
        out = await executor(toolsets, scopes=frozenset()).execute("create_ticket", TICKET)
        assert out.is_error and "permission" in out.content
        assert list_tickets(tickets_db) == []


async def test_per_run_call_limit(tmp_path: Path, searcher) -> None:  # noqa: ANN001
    async with connected(tmp_path) as toolsets:
        ex = executor(toolsets)
        args = {"query": "pods", "progress_title": "x"}
        results = [await ex.execute("search", args) for _ in range(4)]
        assert [r.is_error for r in results] == [False, False, False, True]
        assert "limit" in results[-1].content
        assert len(searcher.queries) == 3


async def test_unknown_tool_and_args_rejected(tmp_path: Path, searcher) -> None:  # noqa: ANN001
    async with connected(tmp_path) as toolsets:
        ex = executor(toolsets)
        assert (await ex.execute("drop_tables", {})).is_error
        r = await ex.execute("search", {"query": "pods", "progress_title": "x", "index": "other"})
        assert r.is_error and "Unknown argument" in r.content
        r = await ex.execute("search", ["not", "an", "object"])
        assert r.is_error
        r = await ex.execute("search", {"query": 5, "progress_title": "x"})
        assert r.is_error and "Invalid arguments" in r.content
        assert searcher.queries == []


async def test_side_effect_held_until_release(tmp_path: Path, tickets_db: str) -> None:
    async with connected(tmp_path) as toolsets:
        gate = Gate()
        task = asyncio.create_task(executor(toolsets, gate=gate).execute("create_ticket", TICKET))
        await asyncio.sleep(0.1)
        assert not task.done() and list_tickets(tickets_db) == []  # held, nothing written
        gate.decide(ALLOW)
        out = await asyncio.wait_for(task, 5)
        assert not out.is_error and len(list_tickets(tickets_db)) == 1


async def test_side_effect_never_runs_when_blocked(tmp_path: Path, tickets_db: str) -> None:
    async with connected(tmp_path) as toolsets:
        gate = Gate()
        task = asyncio.create_task(executor(toolsets, gate=gate).execute("create_ticket", TICKET))
        await asyncio.sleep(0.05)
        gate.decide(DENY)
        out = await asyncio.wait_for(task, 5)
        assert out.is_error
        assert list_tickets(tickets_db) == []


async def test_read_only_tools_not_held(tmp_path: Path) -> None:
    async with connected(tmp_path) as toolsets:
        gate = Gate()  # undecided
        out = await asyncio.wait_for(
            executor(toolsets, gate=gate).execute(
                "search", {"query": "pods", "progress_title": "x"}
            ),
            2,
        )
        assert not out.is_error and out.content.startswith("[1]")
