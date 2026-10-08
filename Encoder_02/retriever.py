"""Retriever interface shared by the UI (and later the evaluation), plus a stub to develop against.

The real bi-encoder only has to implement `Retriever`; nothing that calls it needs to change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from Ingestion_01.doc_dataclass import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float  # higher = more relevant; scale depends on the retriever


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, source: str | None = None) -> list[Hit]:
        """Top-k chunks for `query`, best first; `source` restricts to one document."""
        ...

    def sources(self) -> list[str]:
        """Documents available for filtering."""
        ...


def load_chunks(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(ln)) for ln in f if ln.strip()]


class StubRetriever:
    """Placeholder until the bi-encoder exists: scores a chunk by the share of query words it
    contains. Good enough to exercise the UI with real chunks; NOT a baseline to evaluate."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._words = [self._tokens(c.text) for c in chunks]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> StubRetriever:
        return cls(load_chunks(path))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[Hit]:
        q = self._tokens(query)
        if not q:
            return []
        hits = [Hit(c, len(q & words) / len(q))
                for c, words in zip(self.chunks, self._words)
                if source is None or c.metadata.get("source") == source]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: h.score, reverse=True)  # stable: ties keep document order
        return hits[:k]

    def sources(self) -> list[str]:
        return sorted({c.metadata.get("source", "") for c in self.chunks} - {""})


class BiEncoderRetriever:
    """Dense retrieval: embed the query, nearest neighbours in Chroma, score = cosine similarity.

    `scope` limits searches to some documents (e.g. the book being evaluated) even when the
    shared index also holds others.
    """

    def __init__(self, index, scope: list[str] | None = None):
        self.index = index  # Encoder_02.indexer.ChromaIndex
        self.scope = sorted(scope) if scope else None

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], model_name: str = "BAAI/bge-small-en-v1.5",
                    db_path: str | Path = "data/chroma", device: str = "auto", log=print) -> BiEncoderRetriever:
        """Open the model's collection, sync it with `chunks` (embeds only what is missing), and
        search only their documents."""
        from .embedder import HFEmbedder
        from .indexer import ChromaIndex

        index = ChromaIndex(HFEmbedder(model_name, device=device), db_path)
        stats = index.sync(chunks, log=log)
        log(f"{model_name}: index synced {stats}")
        return cls(index, scope=sorted({c.metadata.get("source", "") for c in chunks} - {""}))

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[Hit]:
        if not query.strip():
            return []
        if source is not None:
            where = {"source": source}
        elif self.scope:
            where = {"source": {"$in": self.scope}}
        else:
            where = None
        return [Hit(chunk, score) for chunk, score in self.index.query(query, k, where)]

    def sources(self) -> list[str]:
        return self.scope or self.index.sources()
