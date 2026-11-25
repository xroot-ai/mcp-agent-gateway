"""App factory. Startup fails loudly if config, prompts or any MCP capability is broken."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from gateway.agents.guardrails import AnthropicGuardrails, Guardrails
from gateway.agents.llm import LLM, AnthropicLLM
from gateway.agents.offline import OfflineGuardrails, OfflineLLM
from gateway.agents.orchestrator import Orchestrator
from gateway.api import chat, history
from gateway.api.errors import install_error_handlers
from gateway.config.loader import load_agent_config, read_prompt
from gateway.config.settings import Settings, get_settings
from gateway.observability import metrics
from gateway.observability.metrics import setup_metrics
from gateway.observability.tracing import setup_tracing
from gateway.persistence.repository import Repository
from gateway.runs.manager import RunManager
from gateway.security.auth import AuthProvider, StaticTokenAuth
from gateway.tools.toolsets import ToolsetManager

log = logging.getLogger("gateway")


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLM | None = None,
    guardrails: Guardrails | None = None,
    auth: AuthProvider | None = None,
) -> FastAPI:
    """``llm`` / ``guardrails`` / ``auth`` overrides exist for tests and local experiments."""
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = load_agent_config(settings.agent_config)
        prompts = {k: read_prompt(Path(v)) for k, v in config.prompts.model_dump().items()}
        setup_tracing(settings)
        setup_metrics(settings)

        repo = Repository(settings.database_url)
        await repo.init()
        toolsets = ToolsetManager()
        try:
            await toolsets.connect(config)  # startup check: every configured tool must exist
            model: LLM | None = llm
            guard = guardrails
            if settings.offline_llm:
                log.warning(
                    "GW_OFFLINE_LLM is on: no model is called; answers are retrieved passages "
                    "and guardrails are keyword rules. Demo use only."
                )
                model = model or OfflineLLM()
                if config.features.guardrails:
                    guard = guard or OfflineGuardrails()
            orchestrator = Orchestrator(
                model or AnthropicLLM(config.models.orchestrator, config.models.fallbacks),
                prompts["orchestrator"],
                prompts["followups"],
                config.features,
                config.limits,
            )
            if guard is None and config.features.guardrails:
                guard = AnthropicGuardrails(config.models.guardrails, prompts["guardrails"])
            app.state.settings = settings
            app.state.repo = repo
            app.state.auth = auth or StaticTokenAuth(settings.api_tokens)
            app.state.runs = RunManager(config, repo, toolsets, orchestrator, guard)
            if not settings.api_tokens and auth is None:
                log.warning("GW_API_TOKENS is empty: every request will be rejected with 401")
            log.info("agent %r ready with tools %s", config.name, sorted(toolsets.bindings))
            yield
            await app.state.runs.shutdown()
        finally:
            await toolsets.aclose()
            await repo.close()

    app = FastAPI(title="mcp-agent-gateway", version="0.1.0", lifespan=lifespan)
    install_error_handlers(app)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def _http_metrics(request: Request, call_next):  # noqa: ANN001, ANN202
        started = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        attrs = {"method": request.method, "route": route, "status": response.status_code}
        metrics.http_requests.add(1, attrs)
        metrics.http_latency.record(time.perf_counter() - started, attrs)
        return response

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(chat.router)
    app.include_router(history.router)
    return app


def run() -> None:
    import uvicorn

    uvicorn.run("gateway.main:create_app", factory=True, host="0.0.0.0", port=8000)  # noqa: S104


if __name__ == "__main__":
    run()
