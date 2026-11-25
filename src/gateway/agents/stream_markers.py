"""Strip control markers out of a streamed answer without leaking half a marker.

The model is asked to write two optional markers:

    [[TITLE: Short conversation title]]      (first turn only, at the very start)
    [[FOLLOWUPS]]                            (at the end, followed by "- question" lines)

Text arrives in arbitrary chunks, so ``[[TIT`` can end one chunk and ``LE: ...]]`` start
the next. The parser holds back any tail that could still become a marker and releases it
once it is clearly ordinary text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

OPEN = "[["
CLOSE = "]]"
_MAX_MARKER_LEN = 200
_TITLE = re.compile(r"^\[\[\s*TITLE\s*:\s*(.+?)\s*\]\]$", re.IGNORECASE | re.DOTALL)
_FOLLOWUPS = re.compile(r"^\[\[\s*FOLLOWUPS\s*\]\]$", re.IGNORECASE)
_KNOWN_PREFIXES = ("[[TITLE", "[[FOLLOWUPS")


@dataclass
class ParsedChunk:
    text: str = ""
    title: str | None = None


@dataclass
class MarkerParser:
    max_followups: int = 3
    _buffer: str = ""
    _in_followups: bool = False
    _followup_text: str = ""
    _answer_parts: list[str] = field(default_factory=list)
    title: str | None = None

    # ------------------------------------------------------------ public API
    def feed(self, chunk: str) -> ParsedChunk:
        if self._in_followups:
            self._followup_text += chunk
            return ParsedChunk()
        self._buffer += chunk
        return self._drain(final=False)

    def finish(self) -> ParsedChunk:
        if self._in_followups:
            return ParsedChunk()
        return self._drain(final=True)

    @property
    def answer(self) -> str:
        return "".join(self._answer_parts).strip()

    @property
    def followups(self) -> list[str]:
        items = []
        for line in self._followup_text.splitlines():
            line = line.strip()
            if line.startswith(("-", "*")):
                line = line[1:].strip()
            if line:
                items.append(line)
        return items[: self.max_followups]

    # ------------------------------------------------------------ internals
    def _drain(self, *, final: bool) -> ParsedChunk:
        out = ParsedChunk()
        emitted: list[str] = []
        while self._buffer:
            start = self._buffer.find(OPEN)
            if start == -1:
                # A trailing "[" may be the first half of "[[".
                keep = 1 if (self._buffer.endswith("[") and not final) else 0
                emitted.append(self._buffer[: len(self._buffer) - keep])
                self._buffer = self._buffer[len(self._buffer) - keep :]
                break
            emitted.append(self._buffer[:start])
            self._buffer = self._buffer[start:]
            end = self._buffer.find(CLOSE)
            if end == -1:
                if final or not self._could_be_marker(self._buffer):
                    emitted.append(self._buffer[: len(OPEN)])
                    self._buffer = self._buffer[len(OPEN) :]
                    continue
                break  # hold back: marker may complete in a later chunk
            candidate = self._buffer[: end + len(CLOSE)]
            rest = self._buffer[end + len(CLOSE) :]
            if m := _TITLE.match(candidate):
                if self.title is None:
                    self.title = out.title = m.group(1).strip()[:120]
                self._buffer = rest.lstrip("\n") if not "".join(emitted).strip() else rest
            elif _FOLLOWUPS.match(candidate):
                self._in_followups = True
                self._followup_text = rest
                self._buffer = ""
                break
            else:
                emitted.append(self._buffer[: len(OPEN)])
                self._buffer = self._buffer[len(OPEN) :]
        out.text = "".join(emitted)
        if self._in_followups:
            out.text = out.text.rstrip()
        self._answer_parts.append(out.text)
        return out

    @staticmethod
    def _could_be_marker(buffer: str) -> bool:
        if len(buffer) > _MAX_MARKER_LEN:
            return False
        head = buffer.upper()
        return any(p.startswith(head) or head.startswith(p) for p in _KNOWN_PREFIXES)
