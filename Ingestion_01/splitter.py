from __future__ import annotations

import re
from collections import Counter
from .doc_dataclass import Document, Section


HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
# "4", "4.1", "3.5.2", "A.1" followed by a capitalised title. At most two digits per part so
# "2019 Results" is no section number; the capital rules out "32 bits" and "0.4 × (0.01 s)".
# A bare letter is not accepted ("A Human Analogy" is a title, not appendix A).
NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*|[A-Z](?:\.\d{1,2})+)\.?\s+[A-Z\"'(“]")
# pymupdf4llm sometimes renders figure/table captions as headings; they are body text.
CAPTION = re.compile(r"^(figure|table)\s+\d", re.IGNORECASE)


class SectionSplitter:
    """Structural split: one Section per Markdown heading, keeping the heading path.

    pymupdf4llm infers heading levels from font size, which is unreliable:
    - Papers often get "4 Design" and "4.1 Encoder" at the same level, so numbered headings
      take their level from the numbering.
    - Unnumbered headings ("Abstract", chapter titles) keep their Markdown level, shifted to
      line up with the shallowest numbered headings.
    - Textbooks give paragraph headings and sidebars the same font as the deepest numbered
      subsections. Unnumbered headings at that depth are nested under the open numbered
      section instead of closing it.
    """

    def split(self, doc: Document) -> list[Section]:
        offset, max_depth = self._numbering_profile(doc)
        sections: list[Section] = []
        stack: list[tuple[int, str, bool]] = []  # (level, title, numbered)
        buf: list[str] = []
        start_page = 1

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                sections.append(Section([t for _, t, _ in stack], text, start_page))
            buf.clear()

        for page in doc.pages:
            for line in page.text.splitlines():
                m = HEADING.match(line)
                title = self._clean_title(m.group(2)) if m else ""
                if m and title and not CAPTION.match(title):
                    flush()
                    depth = self._numbered_depth(title)
                    if depth:
                        level = depth
                    else:
                        level = max(1, len(m.group(1)) - offset)
                        if level >= max_depth:
                            # Paragraph heading / sidebar: goes below the open numbered section.
                            open_numbered = max((lv for lv, _, num in stack if num), default=0)
                            level = max(level, open_numbered + 1)
                    # Close every open heading at this level or deeper, so missing parent
                    # levels (e.g. "##" with no "#" above it) never nest siblings.
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title, bool(depth)))
                    start_page = page.number
                else:
                    if line.strip() and not any(b.strip() for b in buf):
                        start_page = page.number
                    buf.append(line)
        flush()
        return sections

    @staticmethod
    def _clean_title(raw: str) -> str:
        # Drop HTML tags ("CHAPTER <mark>7</mark>") and bold/italic markers, which pymupdf4llm
        # also emits mid-title ("**_Q2: How does_ X**").
        title = re.sub(r"<[^>]+>", "", raw)
        title = re.sub(r"\*+", "", title)
        title = re.sub(r"(?<!\w)_+|_+(?!\w)", "", title)
        return " ".join(title.split())

    @staticmethod
    def _numbered_depth(title: str) -> int | None:
        m = NUMBERED.match(title)
        return m.group(1).count(".") + 1 if m else None

    def _numbering_profile(self, doc: Document) -> tuple[int, int]:
        """(offset, max_depth): how many '#' the shallowest numbered headings sit below their
        depth, and the deepest numbering used. (0, 0) for documents without numbered headings."""
        levels: dict[int, Counter[int]] = {}
        for page in doc.pages:
            for line in page.text.splitlines():
                m = HEADING.match(line)
                if m and (depth := self._numbered_depth(self._clean_title(m.group(2)))):
                    levels.setdefault(depth, Counter())[len(m.group(1))] += 1
        if not levels:
            return 0, 0
        top = min(levels)
        return levels[top].most_common(1)[0][0] - top, max(levels)
