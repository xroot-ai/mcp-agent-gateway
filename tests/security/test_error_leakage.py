from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from gateway.api.errors import install_error_handlers
from tests.conftest import BOB, auth
from tests.fakes import FakeLLM, Script


def test_unhandled_error_is_generic() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("password=hunter2 at db.internal:5432")

    r = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "Something went wrong."}}
    assert "hunter2" not in r.text


def test_validation_error_does_not_echo_input() -> None:
    app = FastAPI()
    install_error_handlers(app)

    class Body(BaseModel):
        n: int

    @app.post("/v")
    async def v(body: Body) -> None: ...

    r = TestClient(app).post("/v", json={"n": "<script>secret-value</script>"})
    assert r.status_code == 422
    assert "secret-value" not in r.text and "body.n" in r.text


def test_other_users_conversation_is_not_found(client_factory) -> None:  # noqa: ANN001
    client = client_factory(FakeLLM([Script(text=["Answer."])]))
    with client.stream("POST", "/v1/chat/completions", json={"message": "hi"}, headers=auth()) as r:
        r.read()
    conv_id = client.get("/v1/conversations", headers=auth()).json()[0]["id"]

    assert client.get(f"/v1/conversations/{conv_id}/messages", headers=auth(BOB)).status_code == 404
    assert client.delete(f"/v1/conversations/{conv_id}", headers=auth(BOB)).status_code == 404
    r = client.post(
        "/v1/chat/completions",
        json={"message": "hi", "conversation_id": conv_id},
        headers=auth(BOB),
    )
    assert r.status_code == 404
    assert client.get("/v1/conversations", headers=auth(BOB)).json() == []


def test_model_failure_is_reported_generically(client_factory) -> None:  # noqa: ANN001
    class ExplodingLLM(FakeLLM):
        async def stream(self, **kwargs):  # noqa: ANN003, ANN201
            raise RuntimeError("upstream said: api key sk-ant-xyz invalid")
            yield  # pragma: no cover

    client = client_factory(ExplodingLLM([]))
    with client.stream("POST", "/v1/chat/completions", json={"message": "hi"}, headers=auth()) as r:
        body = r.read().decode()
    assert "run.failed" in body and "internal_error" in body
    assert "sk-ant" not in body
