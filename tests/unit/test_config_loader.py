from pathlib import Path

import pytest

from gateway.config.loader import ConfigError, load_agent_config, substitute_env

ROOT = Path(__file__).resolve().parents[2]


def test_env_substitution() -> None:
    env = {"A": "x"}
    assert substitute_env("${env:A}", env) == "x"
    assert substitute_env("${env:B:-dflt}", env) == "dflt"
    assert substitute_env("${env:B:-}", env) == ""
    assert substitute_env({"k": ["${env:A}/y"]}, env) == {"k": ["x/y"]}
    with pytest.raises(ConfigError, match="B"):
        substitute_env("${env:B}", env)


def test_shipped_config_loads() -> None:
    cfg = load_agent_config(ROOT / "configs/agents/rag_assistant.yaml", environ={})
    assert cfg.capabilities[0].url == "local://retrieval"
    ticket = cfg.policy_for("create_ticket")
    assert ticket and ticket.side_effect and ticket.context["requester"] == "user_id"


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


PROMPTS = "prompts: {orchestrator: a, guardrails: b, followups: c}\n"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_agent_config(_write(tmp_path, "name: x\nlimitz: {}\n" + PROMPTS))


def test_duplicate_tool_rejected(tmp_path: Path) -> None:
    body = (
        "name: x\n" + PROMPTS + "capabilities:\n"
        "  - {name: a, url: 'local://a', tools: {search: {}}}\n"
        "  - {name: b, url: 'local://b', tools: {search: {}}}\n"
    )
    with pytest.raises(ConfigError, match="search"):
        load_agent_config(_write(tmp_path, body))


def test_bad_url_scheme_rejected(tmp_path: Path) -> None:
    body = "name: x\n" + PROMPTS + "capabilities:\n  - {name: a, url: 'file:///etc', tools: {}}\n"
    with pytest.raises(ConfigError):
        load_agent_config(_write(tmp_path, body))
