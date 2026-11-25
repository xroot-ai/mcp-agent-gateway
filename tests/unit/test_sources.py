import pytest

from gateway.agents.sources import (
    Passage,
    SourceRegistry,
    parse_passages,
    previous_sources_context,
)


def P(url: str, section: str = "", text: str = "body") -> Passage:  # noqa: N802
    return Passage(title=f"T {url}", url=url, section=section, text=text)


def test_numbering_is_first_seen_and_deduplicated_across_calls() -> None:
    reg = SourceRegistry()
    first = reg.add([P("https://a/"), P("https://b", "S")])
    second = reg.add([P("https://B", "s"), P("https://c")])  # same as b (case/slash-insensitive)
    assert [s.number for s, _ in first] == [1, 2]
    assert [s.number for s, _ in second] == [2, 3]
    assert [s.number for s in reg.all] == [1, 2, 3]


def test_render_for_model_labels_passages_with_numbers() -> None:
    reg = SourceRegistry()
    rendered = reg.render_for_model(reg.add([P("https://a", "Intro", "hello")]))
    assert rendered.startswith("[1] T https://a > Intro\nhello")
    assert reg.render_for_model([]) == "No matching passages were found."


def test_cited_in_returns_only_cited_or_all_when_none_cited() -> None:
    reg = SourceRegistry()
    reg.add([P("https://a"), P("https://b"), P("https://c")])
    assert [s.number for s in reg.cited_in("x [3] y [1] z [9]")] == [1, 3]
    assert [s.number for s in reg.cited_in("no citations")] == [1, 2, 3]


def test_contract_violation_raises() -> None:
    with pytest.raises(ValueError):
        parse_passages({"results": []})
    with pytest.raises(ValueError):
        parse_passages({"passages": [{"title": "x"}]})
    assert parse_passages({"passages": []}) == []


def test_previous_sources_context() -> None:
    reg = SourceRegistry()
    reg.add([P("https://a", "S")])
    ctx = previous_sources_context(reg.all)
    assert "(prev 1) T https://a > S <https://a>" in ctx
    assert previous_sources_context([]) == ""
