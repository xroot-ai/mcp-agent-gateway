"""Retrieval MCP server: one ``search`` tool over the Kubernetes docs index.

Run over HTTP:  python -m mcp_servers.retrieval.server   (needs RETRIEVAL_TOKEN)
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_servers.retrieval.overrides import OverrideError, parse_overrides
from mcp_servers.retrieval.search import QdrantHybridSearcher, Searcher
from mcp_servers.retrieval.settings import RetrievalSettings

log = logging.getLogger("retrieval")


def build_server(
    searcher: Searcher | None = None, settings: RetrievalSettings | None = None
) -> MCPServer:
    settings = settings or RetrievalSettings()
    backend: Searcher = searcher or QdrantHybridSearcher(settings)
    server = MCPServer("retrieval", instructions="Search the Kubernetes documentation.")

    @server.tool(
        description=(
            "Search the Kubernetes documentation and return the most relevant passages, "
            "each with title, url and section. Use focused queries; call again with a "
            "different query if the first results miss."
        )
    )
    async def search(
        query: Annotated[str, Field(min_length=2, max_length=300, description="What to look for.")],
        progress_title: Annotated[
            str,
            Field(
                max_length=80,
                description=(
                    "Short status line shown to the user while searching, "
                    "e.g. 'Looking up Pod lifecycle'."
                ),
            ),
        ],
        progress_detail: Annotated[
            str,
            Field(
                max_length=200,
                description=(
                    "One sentence on what this search is for, shown under the status line."
                ),
            ),
        ] = "",
        reason: Annotated[
            str,
            Field(
                max_length=300,
                description=(
                    "Why this search is needed to answer the question (logged for audit)."
                ),
            ),
        ] = "",
        top_k: Annotated[int, Field(ge=1, le=10, description="Number of passages.")] = 5,
        overrides: Annotated[
            dict[str, Any] | None,
            Field(
                description=(
                    "Optional narrowing: {'section_prefix': 'concepts/workloads', "
                    "'min_score': 0.0-1.0}. "
                    "No other keys are accepted."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        try:
            parsed = parse_overrides(overrides)
        except OverrideError as exc:
            raise ToolError(str(exc)) from exc
        log.info("search query=%r top_k=%d reason=%r", query, top_k, reason)
        passages = await backend.search(query, top_k, parsed)
        return {"passages": [p.as_dict() for p in passages]}

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return server


def http_app(settings: RetrievalSettings | None = None):  # noqa: ANN201
    from mcp_servers.http_auth import BearerAuthMiddleware

    settings = settings or RetrievalSettings()
    app = build_server(settings=settings).streamable_http_app(
        stateless_http=True, host=settings.host
    )
    return BearerAuthMiddleware(app, settings.token, open_paths=frozenset({"/healthz"}))


if __name__ == "__main__":
    import uvicorn

    s = RetrievalSettings()
    logging.basicConfig(level="INFO")
    uvicorn.run(http_app(s), host=s.host, port=s.port)
