"""Embed chunks and keep a persistent Chroma collection in sync with them.

    uv run python -m Encoder_02.indexer data/book_chunks.jsonl
    uv run python -m Encoder_02.indexer data/book_chunks.jsonl --model BAAI/bge-base-en-v1.5 --device cpu
    uv run python -m Encoder_02.indexer --remove NetLLM_paper.pdf

chunks.jsonl stays the source of truth; Chroma is an index that can always be rebuilt from it.
Chunk ids are content hashes, so syncing only embeds chunks that are new or changed.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

from Ingestion_01.doc_dataclass import Chunk

from .embedder import Embedder

DEFAULT_DB = Path("data/chroma")
WRITE_BATCH = 256  # chunks embedded + written per step (Chroma accepts up to ~5k)


def collection_name(model_name: str) -> str:
    """One collection per embedding model: vectors from different models must never mix."""
    return "chunks__" + re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


class ChromaIndex:
    def __init__(self, embedder: Embedder, path: str | Path = DEFAULT_DB, collection: str | None = None):
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            collection or collection_name(embedder.model_name),
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedder.model_name},
            embedding_function=None,  # we always pass vectors; never let Chroma embed on its own
        )
        stored = (self.collection.metadata or {}).get("embedding_model")
        if stored != embedder.model_name:
            raise ValueError(f"collection '{self.collection.name}' holds '{stored}' vectors, "
                             f"not '{embedder.model_name}'")

    # ------------------------------------------------------------------------- write

    def sync(self, chunks: Sequence[Chunk], log: Callable[[str], None] = print) -> dict[str, int]:
        """Make the index hold exactly these chunks for every source they cover.
        Sources not present in `chunks` are left untouched."""
        by_source: dict[str, dict[str, Chunk]] = defaultdict(dict)
        duplicates = 0
        for c in chunks:
            src = c.metadata.get("source", "")
            if c.id in by_source[src]:
                duplicates += 1
                continue
            by_source[src][c.id] = c
        if duplicates:
            log(f"skipped {duplicates} duplicate chunk ids")

        stats = {"added": 0, "deleted": 0, "kept": 0}
        for src, wanted in by_source.items():
            existing = set(self.collection.get(where={"source": src}, include=[])["ids"])
            stale = sorted(existing - wanted.keys())
            missing = [c for cid, c in wanted.items() if cid not in existing]
            for i in range(0, len(stale), WRITE_BATCH):
                self.collection.delete(ids=stale[i:i + WRITE_BATCH])
            for i in range(0, len(missing), WRITE_BATCH):
                batch = missing[i:i + WRITE_BATCH]
                self.collection.add(
                    ids=[c.id for c in batch],
                    embeddings=self.embedder.embed_passages([c.text for c in batch]).tolist(),
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata for c in batch],
                )
                log(f"  {src}: embedded {min(i + WRITE_BATCH, len(missing))}/{len(missing)}")
            stats["added"] += len(missing)
            stats["deleted"] += len(stale)
            stats["kept"] += len(wanted) - len(missing)
        return stats

    def remove_source(self, source: str) -> int:
        ids = self.collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    # -------------------------------------------------------------------------- read

    def query(self, query: str, k: int, where: dict | None = None) -> list[tuple[Chunk, float]]:
        """(chunk, cosine similarity), best first."""
        if k <= 0 or self.collection.count() == 0:
            return []
        r = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query).tolist()], n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [(Chunk(cid, doc, meta or {}), 1.0 - dist)
                for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def sources(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({(m or {}).get("source", "") for m in metas} - {""})

    def count(self) -> int:
        return self.collection.count()


def main() -> None:
    from Encoder_02.retriever import load_chunks

    from .embedder import HFEmbedder

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chunks", nargs="*", type=Path, help="chunks JSONL file(s) from the ingestor")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--remove", metavar="SOURCE", action="append", default=[], help="drop a document from the index")
    args = ap.parse_args()
    if not args.chunks and not args.remove:
        ap.error("give chunks file(s) to index and/or --remove SOURCE")

    t0 = time.time()
    embedder = HFEmbedder(args.model, device=args.device, batch_size=args.batch_size)
    index = ChromaIndex(embedder, args.db)
    print(f"{args.model} on {embedder.device} -> {args.db} / {index.collection.name}")

    for src in args.remove:
        print(f"removed {index.remove_source(src)} chunks of {src}")
    for path in args.chunks:
        stats = index.sync(load_chunks(path))
        print(f"{path}: {stats}")
    print(f"index holds {index.count()} chunks from {index.sources()}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
