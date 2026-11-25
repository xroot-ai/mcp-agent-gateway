import pytest

from gateway.tools.context import RunContext, ToolContextError, hide_context_args, resolve_context

CTX = RunContext("run1", "alice", "conv1", "turn1", frozenset({"tickets:write"}))
SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "requester": {"type": "string"}},
    "required": ["title", "requester"],
}


def test_resolve_context() -> None:
    assert resolve_context({"requester": "user_id", "ref": "conversation_id"}, CTX) == {
        "requester": "alice",
        "ref": "conv1",
    }


@pytest.mark.parametrize("key", ["scopes", "password", "__class__", "lookup"])
def test_unknown_or_non_string_keys_fail_loudly(key: str) -> None:
    with pytest.raises(ToolContextError):
        resolve_context({"x": key}, CTX)


def test_empty_value_fails_loudly() -> None:
    with pytest.raises(ToolContextError):
        resolve_context({"requester": "user_id"}, RunContext("r", "", "c", "t"))


def test_hide_context_args_removes_from_schema_without_mutating() -> None:
    hidden = hide_context_args(SCHEMA, {"requester"})
    assert "requester" not in hidden["properties"]
    assert hidden["required"] == ["title"]
    assert "requester" in SCHEMA["properties"]


def test_hide_unknown_arg_fails() -> None:
    with pytest.raises(ToolContextError):
        hide_context_args(SCHEMA, {"tenant"})
