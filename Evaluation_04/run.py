"""Evaluate retrievers on the generated question set and (optionally) fit runtime confidence.

    uv run python -m Evaluation_04.run --chunks data/book_chunks.jsonl --retrievers bm25 stub
    uv run python -m Evaluation_04.run --chunks data/book_chunks.jsonl --retrievers bm25 --calibrate

The first retriever is the reference the others are compared against (paired bootstrap).
Per-query results are written to --out for error analysis; `Evaluation_04.report` recomputes the
tables (and calibration) from them without re-running retrieval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from Encoder_02.reranker import CrossEncoderReranker, RerankingRetriever
from Encoder_02.retriever import BiEncoderRetriever, Retriever, StubRetriever, load_chunks
from Ingestion_01.doc_dataclass import Chunk

from .baselines import BM25Retriever
from .dataset import load_eval_set
from .evaluate import QueryResult, judge, write_results
from .matching import MODES
from .report import add_report_args, calibrate, print_report

# name -> factory(chunks). Dense entries sync their Chroma collection (data/chroma) with the chunks
# first, embedding only what is missing; add other embedding models the same way.
# "+rerank" entries rerank the base retriever's top RERANK_CANDIDATES with a cross-encoder.
RERANKER = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 50


def _bge_small(chunks: list[Chunk]) -> Retriever:
    return BiEncoderRetriever.from_chunks(chunks, "BAAI/bge-small-en-v1.5")


def _reranked(base: Callable[[list[Chunk]], Retriever]) -> Callable[[list[Chunk]], Retriever]:
    return lambda chunks: RerankingRetriever(base(chunks), CrossEncoderReranker(RERANKER), RERANK_CANDIDATES)


RETRIEVERS: dict[str, Callable[[list[Chunk]], Retriever]] = {
    "bm25": BM25Retriever,
    "stub": StubRetriever,
    "bge-small": _bge_small,
    "bm25+rerank": _reranked(BM25Retriever),
    "bge-small+rerank": _reranked(_bge_small),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=Path("data/eval_questions.jsonl"))
    ap.add_argument("--chunks", type=Path, required=True, help="chunks JSONL of the SAME book the questions come from")
    ap.add_argument("--retrievers", nargs="+", default=["bm25"], choices=sorted(RETRIEVERS))
    ap.add_argument("--threshold", type=float, default=90, help="fuzzy evidence match, 0-100")
    ap.add_argument("--out", type=Path, default=Path("data/eval_results"))
    add_report_args(ap)
    args = ap.parse_args()

    items = load_eval_set(args.eval)
    chunks = load_chunks(args.chunks)
    label_sources = {r["source"] for it in items for r in it.relevant}
    chunk_sources = {c.metadata.get("source") for c in chunks}
    if not label_sources & chunk_sources:
        sys.exit(f"The questions are about {sorted(label_sources)} but the chunks come from "
                 f"{sorted(chunk_sources)}. Ingest the same book first.")
    print(f"{len(items)} questions, {len(chunks)} chunks")

    depth = max(*args.ks, args.mrr_k, args.primary_k)
    modes = list(MODES) if args.mode == "both" else [args.mode]
    results: dict[str, list[QueryResult]] = {}
    for name in args.retrievers:
        retriever = RETRIEVERS[name](chunks)
        results[name] = judge(retriever, items, depth, args.threshold,
                              progress=lambda i, n: print(f"\r  {name}: {i}/{n}", end="", flush=True))
        print()
        write_results(results[name], args.out / f"{name}.jsonl", depth)

    print_report(results, modes, args.ks, args.mrr_k, args.primary_k, by_section=args.ci == "section")
    print(f"\nper-query results: {args.out}/<retriever>.jsonl  (re-report with: python -m Evaluation_04.report {args.out})")

    if args.calibrate:
        calibrate(results, args.out, args.primary_k, modes[0])


if __name__ == "__main__":
    main()
