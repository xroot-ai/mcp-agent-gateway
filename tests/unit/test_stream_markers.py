import pytest

from gateway.agents.stream_markers import MarkerParser

FULL = (
    "[[TITLE: Pod restart policies]]\nPods restart according to restartPolicy [1].\n\n"
    "[[FOLLOWUPS]]\n- How do Jobs retry?\n- What is CrashLoopBackOff?\n"
)


def run(chunks: list[str]) -> tuple[str, MarkerParser, list[str]]:
    p = MarkerParser()
    out, titles = [], []
    for c in chunks:
        r = p.feed(c)
        out.append(r.text)
        if r.title:
            titles.append(r.title)
    out.append(p.finish().text)
    return "".join(out), p, titles


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13, len(FULL)])
def test_any_chunking_gives_same_result(size: int) -> None:
    chunks = [FULL[i : i + size] for i in range(0, len(FULL), size)]
    text, p, titles = run(chunks)
    assert text.strip() == "Pods restart according to restartPolicy [1]."
    assert titles == ["Pod restart policies"]
    assert p.followups == ["How do Jobs retry?", "What is CrashLoopBackOff?"]
    assert "[[" not in text


def test_partial_marker_is_held_back_not_leaked() -> None:
    p = MarkerParser()
    assert p.feed("Hello [[TIT").text == "Hello "
    assert p.feed("LE: X]] world").text == " world"
    assert p.title == "X"


def test_ordinary_double_brackets_pass_through() -> None:
    text, p, _ = run(["Use ", "[[", "not a marker]] here and a[0]"])
    assert text == "Use [[not a marker]] here and a[0]"
    assert p.title is None


def test_unclosed_marker_flushed_at_finish() -> None:
    text, _, _ = run(["answer [[TITLE: never closed"])
    assert text == "answer [[TITLE: never closed"


def test_trailing_single_bracket_held_then_released() -> None:
    p = MarkerParser()
    assert p.feed("see [").text == "see "
    assert p.feed("1]").text == "[1]"


def test_followups_capped_and_answer_excludes_them() -> None:
    text, p, _ = run(["A.\n[[FOLLOWUPS]]\n- a\n- b\n- c\n- d"])
    assert p.followups == ["a", "b", "c"]
    assert p.answer == "A."
    assert text == "A."


def test_only_first_title_is_used() -> None:
    _, p, titles = run(["[[TITLE: One]] x [[TITLE: Two]] y"])
    assert titles == ["One"]
    assert p.answer == "x  y"
