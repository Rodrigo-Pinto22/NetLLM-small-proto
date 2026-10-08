"""How retrieved chunks are shown to the LLM, and how its citations are read back."""

from __future__ import annotations

import re
from typing import Sequence

from Encoder_02.retriever import Hit

SYSTEM_PROMPT = """You are a teaching assistant for a computer networking course.
Answer the student's question using ONLY the numbered book excerpts provided.

Rules:
- Cite the excerpts that support each statement with their numbers in square brackets, e.g. [2] or [1][3].
- If the excerpts do not contain the answer, say clearly that the provided book excerpts do not cover it.
  Do not fill gaps with outside knowledge and do not invent citations.
- Be concise and precise: a short paragraph or a few bullet points. Use the book's terminology.
- Answer in the same language as the question."""


def format_context(hits: Sequence[Hit]) -> str:
    """Numbered excerpts. Chunk text already starts with its heading path."""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h.chunk.metadata
        where = ", ".join(x for x in [m.get("source", ""), f"p. {m['page']}" if "page" in m else ""] if x)
        blocks.append(f"[{i}] ({where})\n{h.chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_messages(question: str, context: Sequence[Hit]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Book excerpts:\n\n{format_context(context)}\n\nQuestion: {question.strip()}"},
    ]


CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def extract_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """(valid source numbers cited, numbers that point to no source). Handles [2], [1][3], [1, 3]."""
    nums = {int(n) for group in CITATION.findall(text) for n in re.split(r"\s*[,;]\s*", group)}
    return sorted(n for n in nums if 1 <= n <= n_sources), sorted(n for n in nums if not 1 <= n <= n_sources)
