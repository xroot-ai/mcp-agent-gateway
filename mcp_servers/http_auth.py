"""ASGI middleware that requires ``Authorization: Bearer <token>`` on every HTTP request.

MCP servers refuse to start over HTTP without a token configured, so a misconfigured
deployment fails closed instead of serving unauthenticated tools.
"""

from __future__ import annotations

import hmac
import json

from starlette.types import ASGIApp, Receive, Scope, Send


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp, token: str, open_paths: frozenset[str] = frozenset()) -> None:
        if not token:
            raise RuntimeError("an MCP bearer token must be configured for HTTP transport")
        self.app = app
        self._expected = f"Bearer {token}".encode()
        self._open = open_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self._open:
            await self.app(scope, receive, send)
            return
        supplied = b""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                supplied = value
                break
        if not hmac.compare_digest(supplied, self._expected):
            body = json.dumps({"error": "unauthorized"}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
