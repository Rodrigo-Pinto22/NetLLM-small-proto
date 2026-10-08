from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Sequence


@dataclass
class EvalItem:
    query: str
    type: str
    evidence: str
    relevant: list[dict] = field(default_factory=list)  # [{"source", "section", "page"}]

    @property
    def section_key(self) -> str:
        r = self.relevant[0] if self.relevant else {}
        return f"{r.get('source', '')}|{r.get('section', '')}"


def load_eval_set(path: str | Path) -> list[EvalItem]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    return [EvalItem(r["query"], r.get("type", ""), r.get("evidence", ""), r.get("relevant", [])) for r in rows]


def group_split(keys: Sequence[Hashable], test_frac: float = 0.3, seed: int = 0) -> tuple[list[int], list[int]]:
    """Train/test indices with whole groups on one side. Questions from the same section are
    near-duplicates, so splitting them across train and test would leak."""
    groups = sorted(set(keys), key=str)
    random.Random(seed).shuffle(groups)
    test_groups = set(groups[: round(len(groups) * test_frac)])
    train = [i for i, k in enumerate(keys) if k not in test_groups]
    test = [i for i, k in enumerate(keys) if k in test_groups]
    return train, test
