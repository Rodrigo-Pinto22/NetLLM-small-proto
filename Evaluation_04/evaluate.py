"""Run a retriever over the evaluation set and summarise how well it ranks the right passages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

from Encoder_02.retriever import Retriever

from .dataset import EvalItem
from .matching import MODES, is_relevant
from .metrics import bootstrap_ci, paired_bootstrap, recall_at_k, reciprocal_rank


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


def summarize(results: Sequence[QueryResult], mode: str, metrics: Sequence[str],
              n_boot: int = 2000) -> dict[str, tuple[float, float, float]]:
    """metric -> (mean, CI low, CI high)."""
    return {m: bootstrap_ci(metric_values(results, mode, m), n_boot) for m in metrics}


def summarize_by_type(results: Sequence[QueryResult], mode: str, metric: str,
                      n_boot: int = 2000) -> dict[str, tuple[int, float, float, float]]:
    """question type -> (n, mean, CI low, CI high)."""
    groups: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        groups[r.type].append(r)
    return {t: (len(rs), *bootstrap_ci(metric_values(rs, mode, metric), n_boot)) for t, rs in sorted(groups.items())}


def compare(a: Sequence[QueryResult], b: Sequence[QueryResult], mode: str, metric: str,
            n_boot: int = 2000) -> dict[str, float]:
    """Paired bootstrap of A - B; both must be judged on the same questions in the same order."""
    if [r.query for r in a] != [r.query for r in b]:
        raise ValueError("results are not on the same questions")
    return paired_bootstrap(metric_values(a, mode, metric), metric_values(b, mode, metric), n_boot)
