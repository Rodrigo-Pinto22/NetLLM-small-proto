from __future__ import annotations

import re
from collections import Counter
from .doc_dataclass import Document, Section


HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
# "4", "4.1", "3.5.2", "A", "A.1" before the title. At most two digits so "2019 Results" is not
# a section number. A heading starting with a lone capital ("A Brief History") counts as level 1.
NUMBERED = re.compile(r"^((?:\d{1,2}|[A-Z])(?:\.\d{1,2})*)\.?\s+\S")


class SectionSplitter:
    """Structural split: one Section per Markdown heading, keeping the heading path.

    pymupdf4llm infers heading levels from font size, so papers often get "4 Design" and
    "4.1 Encoder" at the same level. Numbered headings therefore take their level from the
    numbering; unnumbered ones ("Abstract", "References") keep their Markdown level, shifted
    to line up with the numbered top-level sections.
    """

    def split(self, doc: Document) -> list[Section]:
        offset = self._level_offset(doc)
        sections: list[Section] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        buf: list[str] = []
        start_page = 1

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                sections.append(Section([t for _, t in stack], text, start_page))
            buf.clear()

        for page in doc.pages:
            for line in page.text.splitlines():
                m = HEADING.match(line)
                if m:
                    flush()
                    title = self._clean_title(m.group(2))
                    depth = self._numbered_depth(title)
                    level = depth or max(1, len(m.group(1)) - offset)
                    # Close every open heading at this level or deeper, so missing parent
                    # levels (e.g. "##" with no "#" above it) never nest siblings.
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                    start_page = page.number
                else:
                    if line.strip() and not any(b.strip() for b in buf):
                        start_page = page.number
                    buf.append(line)
        flush()
        return sections

    @staticmethod
    def _clean_title(raw: str) -> str:
        # Drop bold/italic markers, which pymupdf4llm also emits mid-title ("**_Q2: How does_ X**").
        title = re.sub(r"\*+", "", raw)
        title = re.sub(r"(?<!\w)_+|_+(?!\w)", "", title)
        return " ".join(title.split())

    @staticmethod
    def _numbered_depth(title: str) -> int | None:
        m = NUMBERED.match(title)
        return m.group(1).count(".") + 1 if m else None

    def _level_offset(self, doc: Document) -> int:
        """How many '#' deeper than level 1 the numbered top-level sections are written."""
        top_levels: Counter[int] = Counter()
        for page in doc.pages:
            for line in page.text.splitlines():
                m = HEADING.match(line)
                if m and self._numbered_depth(self._clean_title(m.group(2))) == 1:
                    top_levels[len(m.group(1))] += 1
        return top_levels.most_common(1)[0][0] - 1 if top_levels else 0
