"""The model can narrow a search but never redirect it to another index or backend."""

import pytest
from mcp import Client

from mcp_servers.retrieval.server import build_server
from tests.fakes import FakeSearcher

BASE = {"query": "pods", "progress_title": "Searching"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"collection": "hr_records"},
        {"qdrant_url": "http://attacker:6333"},
        {"api_key": "x"},
        {"section_prefix": "../../etc"},
        {"min_score": 7},
    ],
)
async def test_override_rejected_and_backend_not_called(overrides: dict) -> None:
    searcher = FakeSearcher()
    async with Client(build_server(searcher=searcher)) as c:
        r = await c.call_tool("search", {**BASE, "overrides": overrides})
    assert r.is_error
    assert searcher.queries == []


@pytest.mark.parametrize(
    "args",
    [
        {**BASE, "top_k": 500},
        {**BASE, "query": "x" * 5000},
        {"query": "pods"},  # progress_title is required
    ],
)
async def test_argument_bounds_enforced(args: dict) -> None:
    searcher = FakeSearcher()
    async with Client(build_server(searcher=searcher)) as c:
        r = await c.call_tool("search", args)
    assert r.is_error
    assert searcher.queries == []


async def test_allowed_override_reaches_backend() -> None:
    searcher = FakeSearcher()
    async with Client(build_server(searcher=searcher)) as c:
        r = await c.call_tool(
            "search", {**BASE, "top_k": 1, "overrides": {"section_prefix": "concepts/workloads"}}
        )
    assert not r.is_error
    ((_, top_k, ov),) = searcher.queries
    assert top_k == 1 and ov.section_prefix == "concepts/workloads"
