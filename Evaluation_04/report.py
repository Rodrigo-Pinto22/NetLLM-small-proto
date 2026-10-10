"""Metrics tables (and confidence calibration) from saved per-query results, without re-running retrieval.

    uv run python -m Evaluation_04.report data/eval_results/bge-small.jsonl data/eval_results/bm25.jsonl
    uv run python -m Evaluation_04.report data/eval_results --calibrate      # every results file in a folder

The first file is the reference the others are compared against. Results files are written by
`Evaluation_04.run`; their retrieval depth (20 by default) is the largest usable k.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .confidence import fit_confidence
from .evaluate import QueryResult, compare, load_results, results_depth, summarize, summarize_by_type
from .matching import MODES


def fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.2f},{hi:.2f}]"


def print_report(results: dict[str, list[QueryResult]], modes: Sequence[str], ks: Sequence[int], mrr_k: int,
                 primary_k: int, by_section: bool = True) -> None:
    metrics = [f"recall@{k}" for k in ks] + [f"mrr@{mrr_k}"]
    names = list(results)
    width = max(len(n) for n in [*names, "retriever"]) + 2
    first = next(iter(results.values()))
    ci = (f"95% CI by section, {len({r.group for r in first})} sections" if by_section else "95% CI by question")
    for mode in modes:
        print(f"\n== {mode} relevance  (n={len(first)}, mean [{ci}]) ==")
        print("".join(f"{h:<{width if i == 0 else 22}}" for i, h in enumerate(["retriever", *metrics])))
        for name, res in results.items():
            s = summarize(res, mode, metrics, by_section=by_section)
            print(f"{name:<{width}}" + "".join(f"{fmt(*s[m]):<22}" for m in metrics))

        primary = f"recall@{primary_k}"
        print(f"\n   by question type ({primary}):")
        for name, res in results.items():
            cells = [f"{t} {fmt(*v[1:])} n={v[0]}"
                     for t, v in summarize_by_type(res, mode, primary, by_section=by_section).items()]
            print(f"   {name:<{width}}" + "   ".join(cells))

        ref = names[0]
        for other in names[1:]:
            c = compare(results[other], results[ref], mode, primary, by_section=by_section)
            verdict = "significant" if c["p"] < 0.05 else "not significant"
            print(f"\n   {other} - {ref} ({primary}): {c['diff']:+.3f} [{c['lo']:+.2f},{c['hi']:+.2f}] "
                  f"p={c['p']:.3f} -> {verdict}")


def calibrate(results: dict[str, list[QueryResult]], out: Path, k: int, mode: str) -> None:
    """Fit and save a runtime ConfidenceModel per retriever (<out>/<name>.confidence.json)."""
    print("\n== runtime confidence ==")
    for name, res in results.items():
        model = fit_confidence(res, name, k=k, mode=mode)
        path = out / f"{name}.confidence.json"
        model.save(path)
        r = model.report
        print(f"{name}: signal={model.signal}  held-out AUROC={r['test_auroc']}  Brier={r['test_brier']}  "
              f"ECE={r['test_ece']}  (train AUROC per signal {r['train_auroc']}) -> {path}")


def add_report_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--mode", choices=[*MODES, "both"], default="both", help="relevance definition")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--mrr-k", type=int, default=10)
    ap.add_argument("--primary-k", type=int, default=5, help="k used for per-type breakdown, comparisons and confidence")
    ap.add_argument("--ci", choices=["section", "question"], default="section",
                    help="resample whole sections (correct for clustered questions) or single questions")
    ap.add_argument("--calibrate", action="store_true", help="fit a runtime ConfidenceModel per retriever")


def results_files(paths: Sequence[Path]) -> list[Path]:
    """Files as given; a folder expands to its *.jsonl files (sorted)."""
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path, help="results .jsonl files (or folders) from Evaluation_04.run")
    ap.add_argument("--out", type=Path, help="where --calibrate writes models (default: next to the first file)")
    add_report_args(ap)
    args = ap.parse_args()

    files = results_files(args.results)
    if not files:
        sys.exit("no results files found")
    results = {f.stem: load_results(f) for f in files}

    depth = min(results_depth(f) for f in files)
    if max(*args.ks, args.mrr_k, args.primary_k) > depth:
        sys.exit(f"results were retrieved to depth {depth}; use --ks/--mrr-k/--primary-k <= {depth}")
    queries = [r.query for r in next(iter(results.values()))]
    for name, res in results.items():
        if [r.query for r in res] != queries:
            sys.exit(f"{name} was evaluated on different questions than {files[0].stem}; re-run them together")

    modes = list(MODES) if args.mode == "both" else [args.mode]
    print(f"{len(queries)} questions, retrievers: {', '.join(results)}")
    print_report(results, modes, args.ks, args.mrr_k, args.primary_k, by_section=args.ci == "section")
    if args.calibrate:
        calibrate(results, args.out or files[0].parent, args.primary_k, modes[0])


if __name__ == "__main__":
    main()
