from __future__ import annotations

from typing import Protocol
from .doc_dataclass import Document, Page
from pathlib import Path

# One parser per format. To support images, PCAP summaries, etc., add a class with
# a parse(path) -> Document method and register it in DEFAULT_PARSERS.

class Parser(Protocol):
    def parse(self, path: Path) -> Document: ...


class PdfParser:
    """PDF -> Markdown per page (keeps headings, which the splitter relies on)."""

    def parse(self, path: Path) -> Document:
        import pymupdf4llm

        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        return Document(
            source=path.name,
            pages=[Page(p["text"], p.get("metadata", {}).get("page", i))
                   for i, p in enumerate(pages, 1)],
        )


class TextParser:
    """Plain text / Markdown / RFC .txt files."""

    def parse(self, path: Path) -> Document:
        return Document(source=path.name,
                        pages=[Page(path.read_text(encoding="utf-8", errors="ignore"), 1)])