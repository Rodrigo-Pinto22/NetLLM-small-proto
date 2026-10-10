"""Run a retriever over the evaluation set and summarise how well it ranks the right passages."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from Encoder_02.retriever import Retriever

from .dataset import EvalItem
from .matching import MODES, is_relevant
from .metrics import bootstrap_ci, first_hit_rank, paired_bootstrap, recall_at_k, reciprocal_rank


@dataclass
class QueryResult:
    query: str
    type: str
    group: str                    # labelled section, for leakage-free splits
    scores: list[float]           # retriever scores, best first
    hits: dict[str, list[bool]]   # mode -> relevance of each retrieved chunk
    retrieved: list[dict]         # metadata of retrieved chunks, for error analysis


def judge(retriever: Retriever, items: Sequence[EvalItem], depth: int = 20, threshold: float = 90,
          progress: Callable[[int, int], None] | None = None) -> list[QueryResult]:
    """Retrieve `depth` chunks per question and mark each relevant or not, under every mode."""
    results = []
    for i, item in enumerate(items, 1):
        hits = retriever.search(item.query, k=depth)
        results.append(QueryResult(
            query=item.query, type=item.type, group=item.section_key,
            scores=[h.score for h in hits],
            hits={m: [is_relevant(h.chunk, item, m, threshold) for h in hits] for m in MODES},
            retrieved=[{"id": h.chunk.id, **h.chunk.metadata} for h in hits],
        ))
        if progress:
            progress(i, len(items))
    return results


def metric_values(results: Sequence[QueryResult], mode: str, metric: str) -> list[float]:
    """Per-query values of 'recall@K' or 'mrr@K'."""
    name, k = metric.split("@")
    fn = {"recall": recall_at_k, "mrr": reciprocal_rank}[name]
    return [fn(r.hits[mode], int(k)) for r in results]


def _groups(results: Sequence[QueryResult], by_section: bool) -> list[str] | None:
    return [r.group for r in results] if by_section else None


def summarize(results: Sequence[QueryResult], mode: str, metrics: Sequence[str],
              n_boot: int = 2000, by_section: bool = True) -> dict[str, tuple[float, float, float]]:
    """metric -> (mean, CI low, CI high). by_section: resample whole sections (cluster bootstrap)."""
    groups = _groups(results, by_section)
    return {m: bootstrap_ci(metric_values(results, mode, m), n_boot, groups=groups) for m in metrics}


def summarize_by_type(results: Sequence[QueryResult], mode: str, metric: str, n_boot: int = 2000,
                      by_section: bool = True) -> dict[str, tuple[int, float, float, float]]:
    """question type -> (n, mean, CI low, CI high)."""
    by_type: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_type[r.type].append(r)
    return {t: (len(rs), *bootstrap_ci(metric_values(rs, mode, metric), n_boot, groups=_groups(rs, by_section)))
            for t, rs in sorted(by_type.items())}


def compare(a: Sequence[QueryResult], b: Sequence[QueryResult], mode: str, metric: str,
            n_boot: int = 2000, by_section: bool = True) -> dict[str, float]:
    """Paired bootstrap of A - B; both must be judged on the same questions in the same order."""
    if [r.query for r in a] != [r.query for r in b]:
        raise ValueError("results are not on the same questions")
    return paired_bootstrap(metric_values(a, mode, metric), metric_values(b, mode, metric), n_boot,
                            groups=_groups(a, by_section))


# --------------------------------------------------------------------- saved results

def write_results(results: Sequence[QueryResult], path: str | Path, depth: int | None = None) -> None:
    """One JSON line per question; `load_results` reads it back (metrics need no re-run).
    `depth` = how many chunks were requested per question (some questions may get fewer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = asdict(r)
            row["first_hit"] = {m: first_hit_rank(r.hits[m]) for m in r.hits}  # for error analysis
            if depth is not None:
                row["depth"] = depth
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def results_depth(path: str | Path) -> int:
    """Requested retrieval depth of a results file; older files without it: the most chunks any
    question got."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    if rows and "depth" in rows[0]:
        return rows[0]["depth"]
    return max((len(next(iter(r["hits"].values()), [])) for r in rows), default=0)


def load_results(path: str | Path) -> list[QueryResult]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    return [QueryResult(r["query"], r["type"], r["group"], r["scores"], r["hits"], r["retrieved"]) for r in rows]
