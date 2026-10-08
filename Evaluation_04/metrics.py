"""Ranking metrics on per-query relevance lists, plus bootstrap uncertainty.

`hits` is the relevance of each retrieved chunk, best first: [False, True, False, ...].
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def recall_at_k(hits: Sequence[bool], k: int) -> float:
    """1 if a relevant chunk is in the top k. With one relevant passage per question this is
    Recall@k (a.k.a. hit rate)."""
    return float(any(hits[:k]))


def reciprocal_rank(hits: Sequence[bool], k: int = 10) -> float:
    """1/rank of the first relevant chunk within the top k, else 0. Averaged: MRR@k."""
    for i, h in enumerate(hits[:k]):
        if h:
            return 1.0 / (i + 1)
    return 0.0


def first_hit_rank(hits: Sequence[bool]) -> int | None:
    return next((i + 1 for i, h in enumerate(hits) if h), None)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high): percentile bootstrap CI of the mean over queries."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 0) -> dict[str, float]:
    """Is A better than B on the SAME queries? Resamples queries, keeping each pair together.

    Returns the mean difference A - B, its CI and a two-sided p-value.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        raise ValueError("no queries to compare")
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(p)}
