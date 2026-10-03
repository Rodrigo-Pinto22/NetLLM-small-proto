from __future__ import annotations
from .doc_dataclass import Document, Page, Section, Chunk

import re
from collections import Counter

class Cleaner:
    """Strip/trim step: running headers/footers, page numbers, hyphenation, whitespace."""

    def __init__(self, min_repeats: int = 5, repeat_ratio: float = 0.03):
        self.min_repeats = min_repeats
        self.repeat_ratio = repeat_ratio

    def clean(self, doc: Document) -> Document:
        boilerplate = self._repeated_edge_lines(doc.pages)
        pages = [Page(self._clean_text(p.text, boilerplate), p.number) for p in doc.pages]
        return Document(doc.source, [p for p in pages if p.text], doc.metadata)

    @staticmethod
    def _norm(line: str) -> str:
        # Ignore digits so "Chapter 3 • Transport Layer 215" matches across pages.
        return re.sub(r"\d+", "", line).strip(" *_#|-•\t").lower()

    def _repeated_edge_lines(self, pages: list[Page]) -> set[str]:
        """Lines that keep appearing at the top/bottom of pages are headers/footers."""
        if len(pages) < self.min_repeats:
            return set()
        counts: Counter[str] = Counter()
        for p in pages:
            lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
            counts.update({self._norm(ln) for ln in lines[:2] + lines[-2:]})
        threshold = max(self.min_repeats, self.repeat_ratio * len(pages))
        return {ln for ln, n in counts.items() if ln and n >= threshold}

    def _clean_text(self, text: str, boilerplate: set[str]) -> str:
        # Re-join words broken across lines ("connec-\ntion"). Rarely also joins a real
        # line-final hyphen ("end-\nto-end"); acceptable for retrieval.
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        kept = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                kept.append("")
            elif self._norm(s) in boilerplate or re.fullmatch(r"\d{1,4}", s):
                continue  # header/footer or bare page number
            else:
                kept.append(line.rstrip())
        text = "\n".join(kept)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()