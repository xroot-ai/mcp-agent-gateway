"""Heading-aware chunking for Hugo-flavoured Markdown.

Each chunk stays inside one section, carries its heading path, and is split on paragraph
boundaries with a small overlap when a section is long.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SHORTCODE_BLOCK = re.compile(r"\{\{[<%]\s*/?\s*[\w-]+[^}]*[>%]\}\}")
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_ANCHOR = re.compile(r"\s*\{#[\w-]+\}\s*$")
_NAMESPACE = uuid.UUID("6f1d3f0e-2a52-4c1e-9a43-0c7c7d3b9a10")


@dataclass
class Chunk:
    id: str
    url: str
    title: str
    section: str
    text: str
    path_prefixes: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, object]:
        return {
            "url": self.url,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "path_prefixes": self.path_prefixes,
        }


def doc_url(rel_path: Path) -> str:
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] in ("_index", "index"):
        parts = parts[:-1]
    return "https://kubernetes.io/docs/" + "/".join(parts) + ("/" if parts else "")


def prefixes(rel_path: Path) -> list[str]:
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] in ("_index", "index"):
        parts = parts[:-1]
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def clean(body: str) -> str:
    body = _HTML_COMMENT.sub("", body)
    body = _SHORTCODE_BLOCK.sub("", body)
    return body


def split_sections(body: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, text)], skipping headings inside fenced code blocks."""
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    path: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            level = len(m.group(1))
            heading = _ANCHOR.sub("", m.group(2)).strip()
            path = path[: level - 1] + [heading]
            sections.append((list(path), []))
        else:
            sections[-1][1].append(line)
    return [(p, "\n".join(lines).strip()) for p, lines in sections if "\n".join(lines).strip()]


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > max_chars:  # very long paragraph or code block
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap :]
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + para) if tail else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def chunk_file(
    path: Path, root: Path, max_chars: int = 1200, overlap: int = 150, min_chars: int = 80
) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, object] = {}
    if m := _FRONT_MATTER.match(raw):
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        raw = raw[m.end() :]
    rel = path.relative_to(root)
    title = str(meta.get("title") or rel.stem.replace("-", " ").title())
    url = doc_url(rel)
    out = []
    for heading_path, text in split_sections(clean(raw)):
        section = " > ".join(heading_path)
        for i, piece in enumerate(split_text(text, max_chars, overlap)):
            if len(piece) < min_chars:
                continue
            cid = str(uuid.uuid5(_NAMESPACE, f"{rel}#{section}#{i}"))
            out.append(Chunk(cid, url, title, section, piece, prefixes(rel)))
    return out


def chunk_corpus(root: Path, **kwargs: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        chunks.extend(chunk_file(path, root, **kwargs))
    return chunks
