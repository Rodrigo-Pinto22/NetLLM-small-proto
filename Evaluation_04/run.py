"""Evaluate retrievers on the generated question set and (optionally) fit runtime confidence.

    uv run python -m Evaluation_04.run --chunks data/book_chunks.jsonl --retrievers bm25 stub
    uv run python -m Evaluation_04.run --chunks data/book_chunks.jsonl --retrievers bm25 --calibrate

The first retriever is the reference the others are compared against (paired bootstrap).
Per-query results are written to --out for error analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from Encoder_02.retriever import BiEncoderRetriever, Retriever, StubRetriever, load_chunks
from Ingestion_01.doc_dataclass import Chunk

from .baselines import BM25Retriever
from .confidence import fit_confidence
from .dataset import load_eval_set
from .evaluate import QueryResult, compare, judge, summarize, summarize_by_type
from .matching import MODES
from .metrics import first_hit_rank

# name -> factory(chunks). Dense entries sync their Chroma collection (data/chroma) with the chunks
# first, embedding only what is missing; add other embedding models the same way.
RETRIEVERS: dict[str, Callable[[list[Chunk]], Retriever]] = {
    "bm25": BM25Retriever,
    "stub": StubRetriever,
    "bge-small": lambda chunks: BiEncoderRetriever.from_chunks(chunks, "BAAI/bge-small-en-v1.5"),
}


def fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.2f},{hi:.2f}]"


def print_report(results: dict[str, list[QueryResult]], modes: list[str], ks: list[int], mrr_k: int,
                 primary_k: int) -> None:
    metrics = [f"recall@{k}" for k in ks] + [f"mrr@{mrr_k}"]
    names = list(results)
    width = max(len(n) for n in [*names, "retriever"]) + 2
    for mode in modes:
        n = len(next(iter(results.values())))
        print(f"\n== {mode} relevance  (n={n}, mean [95% CI]) ==")
        print("".join(f"{h:<{width if i == 0 else 22}}" for i, h in enumerate(["retriever", *metrics])))
        for name, res in results.items():
            s = summarize(res, mode, metrics)
            print(f"{name:<{width}}" + "".join(f"{fmt(*s[m]):<22}" for m in metrics))

        primary = f"recall@{primary_k}"
        print(f"\n   by question type ({primary}):")
        for name, res in results.items():
            cells = [f"{t} {fmt(*v[1:])} n={v[0]}" for t, v in summarize_by_type(res, mode, primary).items()]
            print(f"   {name:<{width}}" + "   ".join(cells))

        ref = names[0]
        for other in names[1:]:
            c = compare(results[other], results[ref], mode, primary)
            verdict = "significant" if c["p"] < 0.05 else "not significant"
            print(f"\n   {other} - {ref} ({primary}): {c['diff']:+.3f} [{c['lo']:+.2f},{c['hi']:+.2f}] "
                  f"p={c['p']:.3f} -> {verdict}")


def write_results(results: list[QueryResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = asdict(r)
            row["first_hit"] = {m: first_hit_rank(r.hits[m]) for m in r.hits}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=Path("data/eval_questions.jsonl"))
    ap.add_argument("--chunks", type=Path, required=True, help="chunks JSONL of the SAME book the questions come from")
    ap.add_argument("--retrievers", nargs="+", default=["bm25"], choices=sorted(RETRIEVERS))
    ap.add_argument("--mode", choices=[*MODES, "both"], default="both", help="relevance definition")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--mrr-k", type=int, default=10)
    ap.add_argument("--primary-k", type=int, default=5, help="k used for per-type breakdown, comparisons and confidence")
    ap.add_argument("--threshold", type=float, default=90, help="fuzzy evidence match, 0-100")
    ap.add_argument("--out", type=Path, default=Path("data/eval_results"))
    ap.add_argument("--calibrate", action="store_true", help="fit a runtime ConfidenceModel per retriever")
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
        write_results(results[name], args.out / f"{name}.jsonl")

    print_report(results, modes, args.ks, args.mrr_k, args.primary_k)
    print(f"\nper-query results: {args.out}/<retriever>.jsonl")

    if args.calibrate:
        print("\n== runtime confidence ==")
        for name, res in results.items():
            model = fit_confidence(res, name, k=args.primary_k, mode=modes[0])
            path = args.out / f"{name}.confidence.json"
            model.save(path)
            r = model.report
            print(f"{name}: signal={model.signal}  held-out AUROC={r['test_auroc']}  Brier={r['test_brier']}  "
                  f"ECE={r['test_ece']}  (train AUROC per signal {r['train_auroc']}) -> {path}")


if __name__ == "__main__":
    main()
