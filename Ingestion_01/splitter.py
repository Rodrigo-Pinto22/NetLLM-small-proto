from __future__ import annotations

import re
from .doc_dataclass import Document, Section


HEADING = re.compile(r"^(#{1,6})\s+(.+)$")

class SectionSplitter:
    """Structural split: one Section per Markdown heading, keeping the heading path.

    pymupdf4llm infers headings from font size, so levels can be noisy on some books;
    inspect a few sections and adjust if needed.
    """

    def split(self, doc: Document) -> list[Section]:
        sections: list[Section] = []
        stack: list[str] = []
        buf: list[str] = []
        start_page = 1

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                sections.append(Section(list(stack), text, start_page))
            buf.clear()

        for page in doc.pages:
            for line in page.text.splitlines():
                m = HEADING.match(line)
                if m:
                    flush()
                    level, title = len(m.group(1)), m.group(2).strip(" *_")
                    stack[:] = stack[: level - 1] + [title]
                    start_page = page.number
                else:
                    if line.strip() and not any(b.strip() for b in buf):
                        start_page = page.number
                    buf.append(line)
        flush()
        return sections
    
    