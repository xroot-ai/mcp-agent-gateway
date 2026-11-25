from pathlib import Path

from ingest.chunk import chunk_file, doc_url, prefixes, split_text

DOC = """---
title: Pod Lifecycle
weight: 30
---
<!-- overview -->
{{< glossary_tooltip term_id="pod" >}} intro paragraph that is long enough to keep around for sure.

## Container restart policy {#restart-policy}

The restartPolicy applies to all containers in the Pod. Values: Always, OnFailure, Never.

```yaml
# not a heading
restartPolicy: Always
```

### Details

More detail text that is long enough to be its own chunk in the index, honestly.
"""


def test_chunk_file_headings_metadata_and_cleanup(tmp_path: Path) -> None:
    f = tmp_path / "concepts/workloads/pods/pod-lifecycle.md"
    f.parent.mkdir(parents=True)
    f.write_text(DOC)
    chunks = chunk_file(f, tmp_path, min_chars=40)
    assert {c.title for c in chunks} == {"Pod Lifecycle"}
    sections = [c.section for c in chunks]
    assert sections == ["", "Container restart policy", "Container restart policy > Details"]
    assert all("{{" not in c.text and "<!--" not in c.text for c in chunks)
    assert "# not a heading" in chunks[1].text  # heading-like lines in code fences stay put
    assert chunks[0].url == "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/"
    assert "concepts/workloads" in chunks[0].path_prefixes
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunk_file(f, tmp_path, min_chars=40)[0].id == chunks[0].id  # ids are stable


def test_index_pages_and_prefixes() -> None:
    assert doc_url(Path("concepts/_index.md")) == "https://kubernetes.io/docs/concepts/"
    assert prefixes(Path("concepts/storage/volumes.md")) == [
        "concepts",
        "concepts/storage",
        "concepts/storage/volumes",
    ]


def test_split_text_respects_limit_with_overlap() -> None:
    paras = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(6))
    pieces = split_text(paras, max_chars=800, overlap=50)
    assert len(pieces) > 1 and all(len(p) <= 800 for p in pieces)
