"""Process-level settings, read from environment variables (prefix ``GW_``)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GW_", env_file=".env", extra="ignore")

    agent_config: str = "configs/agents/rag_assistant.yaml"
    database_url: str = "sqlite+aiosqlite:///./data/gateway.db"

    # Bearer tokens accepted by the dev auth provider:
    # GW_API_TOKENS='{"<token>": {"user_id": "alice", "scopes": ["tickets:write"]}}'
    api_tokens: dict[str, dict[str, Any]] = Field(default_factory=dict)

    log_level: str = "INFO"
    sse_heartbeat_s: float = 15.0
    cors_origins: list[str] = Field(default_factory=list)

    # Observability (all optional)
    otel_exporter_otlp_endpoint: str | None = None
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    service_name: str = "mcp-agent-gateway"

    @field_validator("api_tokens", mode="before")
    @classmethod
    def _parse_tokens(cls, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value) if value.strip() else {}
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
