import pytest

from mcp_servers.retrieval.overrides import OverrideError, SearchOverrides, parse_overrides


def test_empty_overrides() -> None:
    assert parse_overrides(None) == SearchOverrides()
    assert parse_overrides({}) == SearchOverrides()


def test_allowed_overrides_parse() -> None:
    o = parse_overrides({"section_prefix": "concepts/workloads", "min_score": 0.2})
    assert o == SearchOverrides("concepts/workloads", 0.2)


@pytest.mark.parametrize(
    "raw",
    [
        {"collection": "other_index"},
        {"qdrant_url": "http://evil:6333"},
        {"api_key": "x"},
        {"filter": {"must": []}},
        {"section_prefix": "concepts", "top_k": 100},
    ],
)
def test_unknown_keys_rejected(raw: dict) -> None:
    with pytest.raises(OverrideError, match="not allowed"):
        parse_overrides(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"section_prefix": "../secrets"},
        {"section_prefix": "/etc/passwd"},
        {"section_prefix": "Concepts"},
        {"section_prefix": 3},
        {"min_score": 1.5},
        {"min_score": -0.1},
        {"min_score": True},
        {"min_score": "0.5"},
    ],
)
def test_bad_values_rejected(raw: dict) -> None:
    with pytest.raises(OverrideError):
        parse_overrides(raw)
