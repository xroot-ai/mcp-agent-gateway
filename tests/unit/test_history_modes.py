from gateway.agents.history_modes import PriorTurn, build_history

TURNS = [PriorTurn(f"q{i}", f"a{i}") for i in range(1, 5)]


def test_none_mode() -> None:
    assert build_history(TURNS, "none", 6) == []


def test_questions_only_has_no_answers_and_respects_limit() -> None:
    msgs = build_history(TURNS, "questions_only", 2)
    joined = " ".join(m["content"] for m in msgs)
    assert "q3" in joined and "q4" in joined and "q1" not in joined
    assert not any(a in joined for a in ("a1", "a2", "a3", "a4"))
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_first_and_last() -> None:
    msgs = build_history(TURNS, "first_and_last", 6)
    assert [m["content"] for m in msgs] == ["q1", "a1", "q4", "a4"]
    single = build_history(TURNS[:1], "first_and_last", 6)
    assert [m["content"] for m in single] == ["q1", "a1"]


def test_turns_without_answers_are_skipped() -> None:
    msgs = build_history([PriorTurn("blocked", ""), PriorTurn("ok", "yes")], "first_and_last", 6)
    assert [m["content"] for m in msgs] == ["ok", "yes"]


def test_empty_history() -> None:
    assert build_history([], "questions_only", 6) == []
