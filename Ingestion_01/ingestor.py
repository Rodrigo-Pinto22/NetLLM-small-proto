"""
Used offline to build the knowledge base AND at runtime for files users upload.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from .doc_dataclass import Document, Chunk, Page
from .chunker import Chunker
from .splitter import SectionSplitter
from .cleaner import Cleaner
from .parsers import PdfParser, TextParser, Parser


# ----------------------------------------------------------------------- ingestor
DEFAULT_PARSERS: dict[str, Parser] = {".pdf": PdfParser(), ".md": TextParser(), ".txt": TextParser()}

class Ingestor:
    """load -> clean -> split -> chunk. Each stage is swappable."""

    def __init__(self, parsers: dict[str, Parser] | None = None, cleaner: Cleaner | None = None,
                 splitter: SectionSplitter | None = None, chunker: Chunker | None = None):
        self.parsers = parsers or DEFAULT_PARSERS
        self.cleaner = cleaner or Cleaner()
        self.splitter = splitter or SectionSplitter()
        self.chunker = chunker or Chunker()

    def load(self, path: str | Path) -> Document:
        path = Path(path)
        parser = self.parsers.get(path.suffix.lower())
        if parser is None:
            raise ValueError(f"No parser for '{path.suffix}'. Supported: {sorted(self.parsers)}")
        return parser.parse(path)

    def ingest(self, path: str | Path) -> list[Chunk]:
        """A file on disk: a textbook offline, or a user upload at runtime."""
        return self._process(self.load(path))

    def ingest_text(self, text: str, source: str = "user_input") -> list[Chunk]:
        """Raw text pasted into the chat at runtime."""
        return self._process(Document(source, [Page(text, 1)]))

    def ingest_dir(self, folder: str | Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix.lower() in self.parsers:
                chunks.extend(self.ingest(path))
        return chunks

    def _process(self, doc: Document) -> list[Chunk]:
        doc = self.cleaner.clean(doc)
        return self.chunker.chunk(doc, self.splitter.split(doc))


def write_jsonl(chunks: list[Chunk], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="file or folder")
    ap.add_argument("--out", default="data/chunks.jsonl")
    args = ap.parse_args()

    ing = Ingestor()
    src = Path(args.input)
    chunks = ing.ingest_dir(src) if src.is_dir() else ing.ingest(src)
    write_jsonl(chunks, args.out)

    sizes = sorted(ing.chunker.n_tokens(c.text) for c in chunks)
    if sizes:
        print(f"{len(chunks)} chunks -> {args.out}")
        print(f"tokens  min {sizes[0]}  median {sizes[len(sizes) // 2]}  max {sizes[-1]}")