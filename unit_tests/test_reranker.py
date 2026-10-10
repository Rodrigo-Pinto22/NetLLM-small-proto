import pytest

from Encoder_02.reranker import CrossEncoderReranker, RerankingRetriever
from Encoder_02.retriever import StubRetriever
from Ingestion_01.doc_dataclass import Chunk


def chunk(id, text, source="book.pdf"):
    return Chunk(id, text, {"source": source, "section": "S", "page": 1, "part": 0})


CHUNKS = [
    chunk("a", "tcp congestion window tcp congestion window"),     # best word overlap
    chunk("b", "tcp congestion control halves the window on loss"),
    chunk("c", "after a timeout tcp reno sets the congestion window to one segment"),
    chunk("d", "ospf link state routing", source="paper.pdf"),
]


class FakeReranker:
    """Scores by a fixed per-text preference; records what it was asked."""

    model_name = "fake-reranker"

    def __init__(self, prefer=("timeout", "halves")):
        self.prefer, self.calls = prefer, []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [sum(len(p) for p in self.prefer if p in t) for t in texts]


# -------------------------------------------------------------------------------- wrapper

def test_reranker_reorders_base_candidates():
    base = StubRetriever(CHUNKS)
    assert [h.chunk.id for h in base.search("tcp congestion window", k=3)][0] == "a"
    hits = RerankingRetriever(base, FakeReranker(), candidates=10).search("tcp congestion window", k=3)
    assert [h.chunk.id for h in hits] == ["c", "b", "a"]
    assert [h.score for h in hits] == [7.0, 6.0, 0.0]   # reranker scores, not base scores


def test_candidates_deeper_than_k_and_never_less_than_k():
    reranker = FakeReranker()
    r = RerankingRetriever(StubRetriever(CHUNKS), reranker, candidates=2)
    r.search("tcp congestion window", k=1)
    assert len(reranker.calls[-1][1]) == 2            # looked at 2 candidates to return 1
    assert len(r.search("tcp congestion window", k=3)) == 3   # k > candidates: still k results


def test_recall_is_bounded_by_candidates():
    # With 1 candidate the reranker cannot rescue chunk "c", which the base ranks lower.
    hits = RerankingRetriever(StubRetriever(CHUNKS), FakeReranker(), candidates=1).search("tcp congestion window", k=1)
    assert [h.chunk.id for h in hits] == ["a"]


def test_source_filter_passes_through_and_sources_delegate():
    reranker = FakeReranker()
    r = RerankingRetriever(StubRetriever(CHUNKS), reranker, candidates=10)
    assert [h.chunk.id for h in r.search("ospf routing", source="paper.pdf")] == ["d"]
    assert r.sources() == ["book.pdf", "paper.pdf"]


def test_no_candidates_skips_reranker():
    reranker = FakeReranker()
    assert RerankingRetriever(StubRetriever(CHUNKS), reranker).search("quantum") == []
    assert reranker.calls == []


def test_ties_keep_base_order():
    hits = RerankingRetriever(StubRetriever(CHUNKS), FakeReranker(prefer=()), candidates=10).search("tcp window", k=3)
    assert [h.chunk.id for h in hits] == [h.chunk.id for h in StubRetriever(CHUNKS).search("tcp window", k=3)]


def test_invalid_candidates():
    with pytest.raises(ValueError):
        RerankingRetriever(StubRetriever(CHUNKS), FakeReranker(), candidates=0)


def test_works_with_evaluation_and_registry():
    from Evaluation_04.dataset import EvalItem
    from Evaluation_04.evaluate import judge
    from Evaluation_04.run import RETRIEVERS

    assert {"bm25+rerank", "bge-small+rerank"} <= set(RETRIEVERS)
    item = EvalItem("what does tcp reno do after a timeout", "factual",
                    "after a timeout tcp reno sets the congestion window to one segment",
                    [{"source": "book.pdf", "section": "S", "page": 1}])
    [res] = judge(RerankingRetriever(StubRetriever(CHUNKS), FakeReranker(), 10), [item], depth=2)
    assert res.hits["evidence"][0] and res.retrieved[0]["id"] == "c"


# ---------------------------------------------------------------------------- cross-encoder

@pytest.fixture(scope="module")
def tiny_cross_encoder(tmp_path_factory):
    """Random 2-layer BERT classifier + word vocab saved locally, for 1 and 2 output labels."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    words = "tcp reno congestion window timeout segment ospf link state router what does do after a".split()
    paths = {}
    for labels in (1, 2):
        d = tmp_path_factory.mktemp(f"tiny-ce-{labels}")
        (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
        BertTokenizer(str(d / "vocab.txt")).save_pretrained(d)
        BertForSequenceClassification(BertConfig(
            vocab_size=len(words) + 5, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, num_labels=labels, max_position_embeddings=64)).save_pretrained(d)
        paths[labels] = str(d)
    return paths


@pytest.mark.parametrize("labels", [1, 2])
def test_cross_encoder_scores_are_deterministic_and_batch_invariant(tiny_cross_encoder, labels):
    texts = ["tcp reno timeout", "ospf link state router", "congestion window segment", "a"]
    small = CrossEncoderReranker(tiny_cross_encoder[labels], device="cpu", batch_size=1)
    big = CrossEncoderReranker(tiny_cross_encoder[labels], device="cpu", batch_size=16)
    s1, s2 = small.score("what does tcp do", texts), big.score("what does tcp do", texts)
    assert len(s1) == 4 and all(isinstance(x, float) for x in s1)
    assert s1 == pytest.approx(s2, abs=1e-4)              # padding does not change scores
    assert small.score("what does tcp do", texts) == pytest.approx(s1)
    assert small.score("tcp", []) == []


def test_cross_encoder_truncates_long_passages_not_the_query(tiny_cross_encoder):
    ce = CrossEncoderReranker(tiny_cross_encoder[1], device="cpu", max_length=32)
    long_passage = " ".join(["congestion window"] * 200)     # far beyond max_length / position embeddings
    [score] = ce.score("what does tcp reno do after a timeout", [long_passage])
    assert isinstance(score, float)
