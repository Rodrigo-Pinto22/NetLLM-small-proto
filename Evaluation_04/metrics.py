"""Ranking metrics on per-query relevance lists, plus bootstrap uncertainty.

`hits` is the relevance of each retrieved chunk, best first: [False, True, False, ...].
"""

from __future__ import annotations

from typing import Hashable, Sequence

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


def _boot_means(values: np.ndarray, groups: Sequence[Hashable] | None, n_boot: int,
                rng: np.random.Generator) -> np.ndarray:
    """Means of `n_boot` bootstrap resamples. With `groups`, whole groups are resampled (cluster
    bootstrap): questions about the same section succeed or fail together, so they must not be
    counted as independent evidence."""
    if groups is not None and len(set(groups)) > 1:
        ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
        gid = np.fromiter((ids[g] for g in groups), dtype=int, count=len(groups))
        sums = np.bincount(gid, weights=values, minlength=len(ids))
        counts = np.bincount(gid, minlength=len(ids))
        idx = rng.integers(0, len(ids), (n_boot, len(ids)))
        return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return values[rng.integers(0, values.size, (n_boot, values.size))].mean(axis=1)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 groups: Sequence[Hashable] | None = None) -> tuple[float, float, float]:
    """(mean, low, high): percentile bootstrap CI of the mean over queries.

    Pass `groups` (e.g. each question's section) to resample whole groups instead of queries.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if groups is not None and len(groups) != v.size:
        raise ValueError("groups must have one entry per value")
    means = _boot_means(v, groups, n_boot, np.random.default_rng(seed))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 0, groups: Sequence[Hashable] | None = None) -> dict[str, float]:
    """Is A better than B on the SAME queries? Resamples queries (or whole `groups`), keeping each
    pair together.

    Returns the mean difference A - B, its CI and a two-sided p-value.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        raise ValueError("no queries to compare")
    if groups is not None and len(groups) != d.size:
        raise ValueError("groups must have one entry per value")
    boots = _boot_means(d, groups, n_boot, np.random.default_rng(seed))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(p)}
