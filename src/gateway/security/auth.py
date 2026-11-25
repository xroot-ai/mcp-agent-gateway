"""Bearer-token authentication with a pluggable provider.

The default provider checks a static token table from settings (good for local dev and
demos). Swap in an OIDC/JWT provider by implementing ``AuthProvider``.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol

from fastapi import Depends, Request

from gateway.api.errors import Unauthorized


@dataclass(frozen=True)
class Principal:
    user_id: str
    scopes: frozenset[str] = field(default_factory=frozenset)


class AuthProvider(Protocol):
    async def authenticate(self, token: str) -> Principal | None: ...


class StaticTokenAuth:
    def __init__(self, tokens: dict[str, dict[str, Any]]) -> None:
        self._grants = [
            (tok.encode(), Principal(str(g["user_id"]), frozenset(g.get("scopes", []))))
            for tok, g in tokens.items()
            if tok
        ]

    async def authenticate(self, token: str) -> Principal | None:
        candidate = token.encode()
        found = None
        for expected, principal in self._grants:  # constant-time compare against each
            if hmac.compare_digest(candidate, expected):
                found = principal
        return found


async def require_principal(request: Request) -> Principal:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise Unauthorized()
    provider: AuthProvider = request.app.state.auth
    principal = await provider.authenticate(token.strip())
    if principal is None:
        raise Unauthorized()
    return principal


RequiredPrincipal = Annotated[Principal, Depends(require_principal)]
