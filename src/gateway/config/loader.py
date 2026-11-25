"""Agent configuration: YAML file + ``${env:VAR}`` / ``${env:VAR:-default}`` substitution.

Validation is strict (unknown keys are rejected) so a typo in a limit or a tool name
fails at startup instead of silently falling back to a default.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(_Strict):
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(default=16000, ge=256, le=128000)


class Models(_Strict):
    orchestrator: ModelSpec = ModelSpec()
    guardrails: ModelSpec = ModelSpec(effort="low", max_tokens=1024)
    # Server-side refusal fallback ("default" routes by refusal category); null disables it.
    fallbacks: Literal["default"] | None = "default"


class Limits(_Strict):
    max_tool_rounds: int = Field(default=6, ge=1, le=20)
    run_timeout_s: float = Field(default=120, gt=0)
    guardrails_timeout_s: float = Field(default=8, gt=0)
    max_question_chars: int = Field(default=4000, ge=1)


class HistoryConfig(_Strict):
    mode: Literal["questions_only", "first_and_last", "none"] = "questions_only"
    max_turns: int = Field(default=6, ge=0, le=50)
    include_previous_sources: bool = True


class ToolPolicy(_Strict):
    max_calls: int = Field(default=3, ge=0)
    timeout_s: float = Field(default=20, gt=0)
    side_effect: bool = False
    required_scope: str | None = None
    # Emit a progress event built from the tool call's progress_title / progress_detail args.
    progress: bool = False
    # How to read the result: "passages" registers retrieved passages as numbered sources.
    result_kind: Literal["text", "passages"] = "text"
    # Arguments filled by the gateway, never by the model: {arg_name: context_key}.
    context: dict[str, str] = Field(default_factory=dict)


class Capability(_Strict):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    url: str
    token: str | None = None
    tools: dict[str, ToolPolicy]

    @model_validator(mode="after")
    def _check_url(self) -> Capability:
        if not self.url.startswith(("http://", "https://", "local://")):
            raise ValueError(f"capability {self.name!r}: url must be http(s):// or local://")
        return self


class Features(_Strict):
    guardrails: bool = True
    titles: bool = True
    followups: bool = True


class Prompts(_Strict):
    orchestrator: str
    guardrails: str
    followups: str


class AgentConfig(_Strict):
    name: str
    models: Models = Models()
    limits: Limits = Limits()
    history: HistoryConfig = HistoryConfig()
    capabilities: list[Capability] = Field(default_factory=list)
    features: Features = Features()
    prompts: Prompts

    @model_validator(mode="after")
    def _unique_tools(self) -> AgentConfig:
        seen: dict[str, str] = {}
        for cap in self.capabilities:
            for tool in cap.tools:
                if tool in seen:
                    raise ValueError(
                        f"tool {tool!r} declared by both {seen[tool]!r} and {cap.name!r}"
                    )
                seen[tool] = cap.name
        return self

    def policy_for(self, tool_name: str) -> ToolPolicy | None:
        for cap in self.capabilities:
            if tool_name in cap.tools:
                return cap.tools[tool_name]
        return None


def substitute_env(value: Any, environ: Mapping[str, str] | None = None) -> Any:
    """Recursively replace ``${env:VAR}`` and ``${env:VAR:-default}`` in strings."""
    env = os.environ if environ is None else environ
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set and has no default")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: substitute_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_env(v, env) for v in value]
    return value


def load_agent_config(path: str | Path, environ: dict[str, str] | None = None) -> AgentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        return AgentConfig.model_validate(substitute_env(raw, environ))
    except ConfigError:
        raise
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def read_prompt(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigError(f"prompt file {path} is empty")
    return text
