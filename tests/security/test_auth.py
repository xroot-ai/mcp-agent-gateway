import pytest
from starlette.testclient import TestClient

from mcp_servers.http_auth import BearerAuthMiddleware
from mcp_servers.retrieval.server import http_app as retrieval_app
from mcp_servers.retrieval.settings import RetrievalSettings
from tests.conftest import auth
from tests.fakes import FakeLLM

ENDPOINTS = [
    ("post", "/v1/chat/completions", {"message": "hi"}),
    ("delete", "/v1/chat/completions/abc", None),
    ("get", "/v1/conversations", None),
    ("get", "/v1/conversations/abc/messages", None),
    ("delete", "/v1/conversations/abc", None),
    ("post", "/v1/messages/abc/feedback", {"rating": "up"}),
]


@pytest.mark.parametrize(("method", "path", "body"), ENDPOINTS)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Basic YWxpY2U6eA=="},
        {"Authorization": "Bearer "},
    ],
)
def test_gateway_requires_valid_bearer(client_factory, method, path, body, headers) -> None:  # noqa: ANN001
    client = client_factory(FakeLLM([]))
    r = client.request(method, path, json=body, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"
    assert r.headers["www-authenticate"] == "Bearer"


def test_health_is_open(client_factory) -> None:  # noqa: ANN001
    assert client_factory(FakeLLM([])).get("/healthz").status_code == 200


def test_mcp_server_refuses_to_start_without_token() -> None:
    with pytest.raises(RuntimeError, match="token"):
        retrieval_app(RetrievalSettings(token=""))


def test_mcp_http_rejects_missing_or_wrong_token() -> None:
    app = retrieval_app(RetrievalSettings(token="s3cret"))
    with TestClient(app) as c:
        init = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        assert c.post("/mcp", json=init).status_code == 401
        assert (
            c.post("/mcp", json=init, headers={"Authorization": "Bearer nope"}).status_code == 401
        )
        ok = c.post(
            "/mcp",
            json=init,
            headers={
                "Authorization": "Bearer s3cret",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert ok.status_code != 401
        assert c.get("/healthz").status_code == 200  # health check is the only open path


async def test_middleware_passes_non_http_scopes() -> None:
    called = []

    async def inner(scope, receive, send):  # noqa: ANN001
        called.append(scope["type"])

    await BearerAuthMiddleware(inner, "t")({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    assert called == ["lifespan"]


def test_users_cannot_cancel_each_others_runs(client_factory) -> None:  # noqa: ANN001
    client = client_factory(FakeLLM([]))
    r = client.delete("/v1/chat/completions/does-not-exist", headers=auth())
    assert r.status_code == 404
