import re
import zlib

import numpy as np
import pytest

from Encoder_02.indexer import ChromaIndex, collection_name
from Encoder_02.retriever import BiEncoderRetriever
from Ingestion_01.doc_dataclass import Chunk


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims: texts sharing words get similar vectors. Counts calls."""

    model_name = "fake/bow-64"

    def __init__(self):
        self.embedded = 0

    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(w.encode()) % 64] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_passages(self, texts):
        self.embedded += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.empty((0, 64), np.float32)

    def embed_query(self, text):
        return self._vec(text)


def chunk(id, text, source="book.pdf", section="S"):
    return Chunk(id, text, {"source": source, "section": section, "page": 1, "part": 0})


BOOK = [
    chunk("c1", "TCP Reno halves the congestion window after duplicate acknowledgements"),
    chunk("c2", "OSPF floods link state advertisements to every router"),
    chunk("c3", "Ethernet switches learn MAC addresses by self learning"),
]
PAPER = [chunk("p1", "NetLLM adapts large language models for networking with low rank matrices", source="paper.pdf")]


@pytest.fixture
def index(tmp_path):
    return ChromaIndex(FakeEmbedder(), tmp_path / "chroma")


# ------------------------------------------------------------------------------------ index

def test_collection_name_is_per_model_and_valid():
    assert collection_name("BAAI/bge-small-en-v1.5") == "chunks__baai-bge-small-en-v1-5"
    assert collection_name("BAAI/bge-small-en-v1.5") != collection_name("BAAI/bge-base-en-v1.5")


def test_sync_adds_then_is_idempotent(index):
    assert index.sync(BOOK, log=lambda _: None) == {"added": 3, "deleted": 0, "kept": 0}
    assert index.embedder.embedded == 3
    assert index.sync(BOOK, log=lambda _: None) == {"added": 0, "deleted": 0, "kept": 3}
    assert index.embedder.embedded == 3  # nothing re-embedded
    assert index.count() == 3


def test_sync_replaces_changed_chunks_and_leaves_other_sources_alone(index):
    index.sync(BOOK + PAPER, log=lambda _: None)
    changed = [BOOK[0], chunk("c2b", "OSPF uses Dijkstra on the link state database")]  # c2 edited, c3 gone
    assert index.sync(changed, log=lambda _: None) == {"added": 1, "deleted": 2, "kept": 1}
    assert index.count() == 3
    assert index.sources() == ["book.pdf", "paper.pdf"]


def test_sync_skips_duplicate_ids(index):
    logs = []
    assert index.sync(BOOK + [BOOK[0]], log=logs.append)["added"] == 3
    assert any("duplicate" in m for m in logs)


def test_query_returns_chunks_with_cosine_scores(index):
    index.sync(BOOK, log=lambda _: None)
    results = index.query("which router floods link state advertisements", k=2)
    assert results[0][0] == BOOK[1]  # chunk round-trips with text + metadata
    assert 0 < results[1][1] < results[0][1] <= 1.0 + 1e-6
    assert index.query("anything", k=0) == []


def test_query_on_empty_index(index):
    assert index.query("tcp", k=5) == []


def test_remove_source(index):
    index.sync(BOOK + PAPER, log=lambda _: None)
    assert index.remove_source("paper.pdf") == 1
    assert index.remove_source("paper.pdf") == 0
    assert index.sources() == ["book.pdf"]


def test_refuses_other_models_vectors(tmp_path):
    ChromaIndex(FakeEmbedder(), tmp_path / "db", collection="shared")
    other = FakeEmbedder()
    other.model_name = "fake/other"
    with pytest.raises(ValueError, match="holds 'fake/bow-64' vectors"):
        ChromaIndex(other, tmp_path / "db", collection="shared")


def test_index_persists_on_disk(tmp_path):
    ChromaIndex(FakeEmbedder(), tmp_path / "db").sync(BOOK, log=lambda _: None)
    reopened = ChromaIndex(FakeEmbedder(), tmp_path / "db")
    assert reopened.count() == 3


# -------------------------------------------------------------------------------- retriever

def test_bi_encoder_scope_and_source_filter(index):
    index.sync(BOOK + PAPER, log=lambda _: None)
    scoped = BiEncoderRetriever(index, scope=["book.pdf"])
    assert all(h.chunk.metadata["source"] == "book.pdf" for h in scoped.search("networking models", k=10))
    assert scoped.sources() == ["book.pdf"]

    everything = BiEncoderRetriever(index)
    assert {h.chunk.metadata["source"] for h in everything.search("networking", k=10)} == {"book.pdf", "paper.pdf"}
    assert [h.chunk.id for h in everything.search("networking", k=10, source="paper.pdf")] == ["p1"]
    assert everything.sources() == ["book.pdf", "paper.pdf"]
    assert everything.search("   ") == []


def test_bi_encoder_from_chunks_syncs_and_scopes(tmp_path, monkeypatch):
    import Encoder_02.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod, "HFEmbedder", lambda name, device: FakeEmbedder())
    r = BiEncoderRetriever.from_chunks(BOOK, "fake/bow-64", db_path=tmp_path / "db", log=lambda _: None)
    assert r.scope == ["book.pdf"] and r.index.count() == 3
    assert r.search("MAC address learning switches", k=1)[0].chunk.id == "c3"


# ------------------------------------------------------------------------------- embedder

@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A 2-layer random BERT + word-level vocab saved locally: exercises HFEmbedder offline."""
    from transformers import BertConfig, BertModel, BertTokenizer

    d = tmp_path_factory.mktemp("tiny-bert")
    words = "tcp reno congestion window ospf link state router switch mac address represent this " \
            "sentence for searching relevant passages query passage".split()
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ":", *words]))
    BertTokenizer(str(d / "vocab.txt")).save_pretrained(d)
    BertModel(BertConfig(vocab_size=len(words) + 6, hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=64)).save_pretrained(d)
    return str(d)


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_hf_embedder_unit_vectors_and_batching_invariance(tiny_model, pooling):
    from Encoder_02.embedder import HFEmbedder

    emb = HFEmbedder(tiny_model, device="cpu", batch_size=2, pooling=pooling)
    texts = ["tcp reno", "ospf link state router switch mac address", "congestion window"]
    vecs = emb.embed_passages(texts)
    assert vecs.shape == (3, 32)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-5)
    # Padding must not change a text's vector (attention mask respected).
    alone = emb.embed_passages(["tcp reno"])[0]
    assert np.allclose(alone, vecs[0], atol=1e-5)
    assert emb.embed_passages([]).shape == (0, 32)


def test_hf_embedder_query_instruction(tiny_model):
    from Encoder_02.embedder import HFEmbedder

    emb = HFEmbedder(tiny_model, device="cpu")
    assert not np.allclose(emb.embed_query("tcp reno"), emb.embed_passages(["tcp reno"])[0])
    plain = HFEmbedder(tiny_model, device="cpu", query_instruction="")
    assert np.allclose(plain.embed_query("tcp reno"), plain.embed_passages(["tcp reno"])[0], atol=1e-5)
    with pytest.raises(ValueError):
        HFEmbedder(tiny_model, device="cpu", pooling="max")
