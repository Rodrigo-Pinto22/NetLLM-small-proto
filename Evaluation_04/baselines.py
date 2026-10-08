"""Baseline retrievers implementing the `Retriever` contract (Encoder_02.retriever)."""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

from Encoder_02.retriever import Hit
from Ingestion_01.doc_dataclass import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """BM25 keyword search: the classic baseline a dense retriever has to beat.

    Strong on exact terms (protocol names, field names, RFC numbers), weak on paraphrases.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]
        self._len = []
        for i, c in enumerate(chunks):
            tf = Counter(tokenize(c.text))
            self._len.append(sum(tf.values()))
            for term, n in tf.items():
                self._postings[term].append((i, n))
        self._avg_len = sum(self._len) / len(chunks) if chunks else 0.0
        n_docs = len(chunks)
        self._idf = {t: math.log(1 + (n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc, tf in self._postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self._len[doc] / self._avg_len)
                scores[doc] += idf * tf * (self.k1 + 1) / norm
        if source is not None:
            scores = {d: s for d, s in scores.items() if self.chunks[d].metadata.get("source") == source}
        best = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))  # ties: document order
        return [Hit(self.chunks[d], s) for d, s in best]

    def sources(self) -> list[str]:
        return sorted({c.metadata.get("source", "") for c in self.chunks} - {""})
