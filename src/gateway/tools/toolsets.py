"""Connect to MCP servers named in the agent config.

``http://`` / ``https://`` URLs use Streamable HTTP with an optional bearer token.
``local://<name>`` runs a registered server in-process (no network), which is what
tests and single-process demos use.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent

from gateway.config.loader import AgentConfig, Capability, ConfigError, ToolPolicy
from gateway.tools.context import hide_context_args

log = logging.getLogger(__name__)

LocalFactory = Callable[[], MCPServer]
_LOCAL_SERVERS: dict[str, LocalFactory] = {}


def register_local_server(name: str, factory: LocalFactory) -> None:
    _LOCAL_SERVERS[name] = factory


def _local_factory(name: str) -> LocalFactory:
    if name in _LOCAL_SERVERS:
        return _LOCAL_SERVERS[name]
    if name == "retrieval":
        from mcp_servers.retrieval.server import build_server as retrieval

        return retrieval
    if name == "tickets":
        from mcp_servers.tickets.server import build_server as tickets

        return tickets
    raise ConfigError(f"no local MCP server registered as {name!r}")


@dataclass
class ToolBinding:
    name: str
    capability: str
    policy: ToolPolicy
    description: str
    model_schema: dict[str, Any]
    client: Client

    def anthropic_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.model_schema,
            "eager_input_streaming": True,
        }


@dataclass
class CallResult:
    is_error: bool
    text: str
    data: Any


class ToolsetManager:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.bindings: dict[str, ToolBinding] = {}

    async def connect(self, config: AgentConfig) -> None:
        for cap in config.capabilities:
            client = await self._open(cap)
            listed = {t.name: t for t in (await client.list_tools()).tools}
            for tool_name, policy in cap.tools.items():
                tool = listed.get(tool_name)
                if tool is None:
                    raise ConfigError(
                        f"capability {cap.name!r} does not expose tool {tool_name!r} "
                        f"(server offers: {sorted(listed)})"
                    )
                schema = hide_context_args(dict(tool.input_schema), set(policy.context))
                self.bindings[tool_name] = ToolBinding(
                    name=tool_name,
                    capability=cap.name,
                    policy=policy,
                    description=(tool.description or tool_name).strip(),
                    model_schema=schema,
                    client=client,
                )
            log.info("capability %s connected (%d tools)", cap.name, len(cap.tools))

    async def _open(self, cap: Capability) -> Client:
        if cap.url.startswith("local://"):
            server = _local_factory(cap.url.removeprefix("local://"))()
            return await self._stack.enter_async_context(Client(server))
        headers = {"Authorization": f"Bearer {cap.token}"} if cap.token else {}
        http = await self._stack.enter_async_context(
            httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30.0, read=300.0))
        )
        transport = streamable_http_client(cap.url, http_client=http)
        return await self._stack.enter_async_context(Client(transport))

    def definitions(self) -> list[dict[str, Any]]:
        return [b.anthropic_definition() for b in self.bindings.values()]

    async def call(self, binding: ToolBinding, arguments: dict[str, Any]) -> CallResult:
        result = await binding.client.call_tool(binding.name, arguments)
        text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        data: Any = result.structured_content
        if data is None and text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
        return CallResult(is_error=bool(result.is_error), text=text, data=data)

    async def aclose(self) -> None:
        await self._stack.aclose()
