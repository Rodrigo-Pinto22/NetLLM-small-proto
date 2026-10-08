"""Real-time retrieval confidence: "how likely is it that a relevant passage is in what we show?"

At query time there is no ground truth, only the retriever's scores. Label-free signals are
computed from them, and an isotonic calibrator fitted on the evaluation set maps the most
predictive signal to a probability. A ConfidenceModel belongs to ONE retriever: refit it whenever
the retriever, its model or the chunking changes.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from Encoder_02.retriever import Hit

from .dataset import group_split
from .evaluate import QueryResult
from .metrics import recall_at_k

SIGNALS = ("top1", "margin", "spread")
LEVELS = ((0.8, "high"), (0.5, "medium"), (0.0, "low"))


def signals(scores: Sequence[float]) -> dict[str, float]:
    """top1:   score of the best chunk
    margin: lead of the best chunk over the second (a clear winner vs. several look-alikes)
    spread: lead of the best chunk over the mean of the shown ones"""
    s = sorted(scores, reverse=True)
    if not s:
        return dict.fromkeys(SIGNALS, 0.0)
    return {"top1": s[0], "margin": s[0] - (s[1] if len(s) > 1 else 0.0), "spread": s[0] - sum(s) / len(s)}


def auroc(x: Sequence[float], y: Sequence[bool]) -> float:
    """Probability that a random success scores higher than a random failure (0.5 = useless)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=bool)
    pos, neg = x[y], x[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def brier(p: Sequence[float], y: Sequence[bool]) -> float:
    """Mean squared error of the probabilities (0 = perfect, 0.25 = always saying 50%)."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(p: Sequence[float], y: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error: average gap between predicted and observed success rates."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


@dataclass
class IsotonicCalibrator:
    """Monotone step function signal -> probability (pool-adjacent-violators).

    Each step is Laplace-smoothed so a handful of examples never yields a flat 0% or 100%.
    """
    xs: list[float]  # lower edge of each step
    ys: list[float]  # probability on that step

    @classmethod
    def fit(cls, x: Sequence[float], y: Sequence[bool]) -> IsotonicCalibrator:
        if len(x) == 0:
            raise ValueError("cannot calibrate on zero examples")
        by_x: dict[float, list[float]] = {}  # equal signal values must share one step
        for xv, yv in zip(map(float, x), map(float, y)):
            by_x.setdefault(xv, [0.0, 0])
            by_x[xv][0] += yv
            by_x[xv][1] += 1
        blocks: list[list[float]] = []  # [x_min, successes, count]
        for xv in sorted(by_x):
            blocks.append([xv, *by_x[xv]])
            # Merge while the success rate would go down as the signal goes up.
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
                _, s, n = blocks.pop()
                blocks[-1][1] += s
                blocks[-1][2] += n
        return cls([b[0] for b in blocks], [(b[1] + 1) / (b[2] + 2) for b in blocks])

    def predict(self, x: float) -> float:
        return self.ys[max(0, bisect.bisect_right(self.xs, x) - 1)]


@dataclass
class Confidence:
    probability: float
    level: str
    signals: dict[str, float]


@dataclass
class ConfidenceModel:
    retriever: str
    signal: str
    k: int                       # confidence that a relevant chunk is among the top k shown
    mode: str                    # relevance definition it was fitted with
    calibrator: IsotonicCalibrator
    report: dict = field(default_factory=dict)  # held-out quality: auroc / brier / ece / n

    def assess(self, hits: Sequence[Hit]) -> Confidence:
        sig = signals([h.score for h in hits[: self.k]])
        p = self.calibrator.predict(sig[self.signal])
        return Confidence(p, next(name for cut, name in LEVELS if p >= cut), sig)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceModel:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{**d, "calibrator": IsotonicCalibrator(**d["calibrator"])})


def fit_confidence(results: Sequence[QueryResult], retriever: str, k: int = 5, mode: str = "evidence",
                   test_frac: float = 0.3, seed: int = 0) -> ConfidenceModel:
    """Pick the signal that best separates successes from failures, check the calibration on
    held-out sections, then refit on everything for use at runtime."""
    feats = [signals(r.scores[:k]) for r in results]
    labels = [bool(recall_at_k(r.hits[mode], k)) for r in results]
    train, test = group_split([r.group for r in results], test_frac, seed)
    if not train or not test:
        raise ValueError("need questions from several sections to fit and check confidence")

    def col(name: str, idx: list[int]) -> list[float]:
        return [feats[i][name] for i in idx]

    y_train, y_test = [labels[i] for i in train], [labels[i] for i in test]
    train_auc = {s: auroc(col(s, train), y_train) for s in SIGNALS}
    signal = max(SIGNALS, key=lambda s: -1 if np.isnan(train_auc[s]) else train_auc[s])

    held_out = IsotonicCalibrator.fit(col(signal, train), y_train)
    p_test = [held_out.predict(v) for v in col(signal, test)]
    report = {
        "n_train": len(train), "n_test": len(test), "success_rate": float(np.mean(labels)),
        "train_auroc": {s: round(v, 3) for s, v in train_auc.items()},
        "test_auroc": round(auroc(col(signal, test), y_test), 3),
        "test_brier": round(brier(p_test, y_test), 3), "test_ece": round(ece(p_test, y_test), 3),
    }
    final = IsotonicCalibrator.fit(col(signal, list(range(len(results)))), labels)
    return ConfidenceModel(retriever, signal, k, mode, final, report)
