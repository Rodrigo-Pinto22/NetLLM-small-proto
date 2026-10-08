"""When does a piece of text count as "the passage that answers the question"?

Shared by the dataset creation (is the LLM's evidence really in the section?) and the metrics
(is this retrieved chunk relevant?), so both apply exactly the same rule.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from Ingestion_01.doc_dataclass import Chunk

from .dataset import EvalItem

MODES = ("evidence", "section")


def norm(text: str) -> str:
    """Normalise for comparison only (never for stored text): LLMs swap in unicode dashes and
    quotes and drop Markdown markers."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
    text = re.sub(r"[*_`]", "", text)
    return " ".join(text.split()).lower()


def evidence_ok(evidence: str, text: str, threshold: float = 90) -> bool:
    """The evidence sentence (almost) literally occurs in `text`."""
    ev = norm(evidence)
    return len(ev) >= 20 and fuzz.partial_ratio(ev, norm(text)) >= threshold


def is_relevant(chunk: Chunk, item: EvalItem, mode: str = "evidence", threshold: float = 90) -> bool:
    """evidence: the chunk contains the question's evidence sentence (strict; survives re-chunking).
    section: the chunk comes from the labelled section or one of its sub-sections (lenient)."""
    if mode == "evidence":
        return evidence_ok(item.evidence, chunk.text, threshold)
    if mode == "section":
        src, sec = chunk.metadata.get("source"), chunk.metadata.get("section", "")
        return any(src == r["source"] and (sec == r["section"] or sec.startswith(r["section"] + " > "))
                   for r in item.relevant)
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
