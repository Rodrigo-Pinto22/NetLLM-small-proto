import sys
import types

import pytest


class FakeTokenizer:
    """One token per whitespace-separated word: deterministic and needs no model download."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


@pytest.fixture
def fake_transformers(monkeypatch):
    """Replace `transformers` so Chunker() builds a FakeTokenizer instead of downloading bge."""
    requested: list[str] = []

    def from_pretrained(name: str) -> FakeTokenizer:
        requested.append(name)
        return FakeTokenizer()

    module = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", module)
    return requested


@pytest.fixture
def make_chunker(fake_transformers):
    from Ingestion_01.chunker import Chunker

    def make(**kwargs) -> Chunker:
        return Chunker(**kwargs)

    return make


def words(n: int, word: str = "w") -> str:
    return " ".join(f"{word}{i}" for i in range(n))
