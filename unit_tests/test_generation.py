import pytest

from Encoder_02.retriever import Hit, StubRetriever
from Evaluation_04.confidence import ConfidenceModel, IsotonicCalibrator
from Generation_05.answerer import LOW_CONFIDENCE, NO_HITS, RAGAnswerer
from Generation_05.prompt import SYSTEM_PROMPT, build_messages, extract_citations, format_context
from Ingestion_01.doc_dataclass import Chunk


def chunk(id, text, page=1, source="book.pdf", section="3 Transport > 3.7 Congestion"):
    return Chunk(id, f"{section}\n\n{text}", {"source": source, "section": section, "page": page, "part": 0})


CHUNKS = [
    chunk("reno", "After a timeout TCP Reno sets the congestion window to one segment.", page=274),
    chunk("fast", "Fast recovery halves the congestion window after three duplicate ACKs.", page=275),
    chunk("ospf", "OSPF floods link-state advertisements.", page=415, section="4 Network > 4.6.2 OSPF"),
]


class FakeLLM:
    name = "fake-llm"

    def __init__(self, pieces=("TCP Reno resets ", "cwnd to 1 MSS [1]."), error=None):
        self.pieces, self.error, self.calls = pieces, error, []

    def stream(self, messages):
        self.calls.append(messages)
        for p in self.pieces:
            yield p
        if self.error:
            raise self.error


def confidence_model(level_for_any_score: float):
    """Calibrator that returns the same probability for every score."""
    return ConfidenceModel("stub", "top1", 3, "evidence", IsotonicCalibrator([float("-inf")], [level_for_any_score]))


# ----------------------------------------------------------------------------------- prompt

def test_context_is_numbered_with_source_and_page():
    ctx = format_context([Hit(CHUNKS[0], 0.9), Hit(CHUNKS[2], 0.5)])
    assert ctx.startswith("[1] (book.pdf, p. 274)\n3 Transport > 3.7 Congestion\n\nAfter a timeout")
    assert "\n\n[2] (book.pdf, p. 415)\n4 Network > 4.6.2 OSPF" in ctx


def test_messages_have_system_rules_and_question():
    msgs = build_messages("  Why? ", [Hit(CHUNKS[0], 0.9)])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "ONLY the numbered book excerpts" in SYSTEM_PROMPT
    assert msgs[1]["role"] == "user" and msgs[1]["content"].endswith("Question: Why?")
    assert "[1] (book.pdf, p. 274)" in msgs[1]["content"]


@pytest.mark.parametrize("text, valid, invalid", [
    ("Reno resets cwnd [1]. Fast recovery halves it [2].", [1, 2], []),
    ("Both [1][3] and [2, 3] and [1; 2].", [1, 2, 3], []),
    ("Hallucinated [7] and [0].", [], [0, 7]),
    ("No citations at all. Array a[i] is not one.", [], []),
])
def test_extract_citations(text, valid, invalid):
    assert extract_citations(text, n_sources=3) == (valid, invalid)


# --------------------------------------------------------------------------------- answerer

def test_stream_yields_sources_first_then_growing_answer_then_citations():
    llm = FakeLLM()
    updates = list(RAGAnswerer(StubRetriever(CHUNKS), llm).stream("what does reno do after a timeout", k=2))

    first, last = updates[0], updates[-1]
    assert first.text == "" and not first.done and [h.chunk.id for h in first.hits][0] == "reno"
    assert [u.text for u in updates[1:-1]] == ["TCP Reno resets ", "TCP Reno resets cwnd to 1 MSS [1]."]
    assert last.done and last.llm_used and last.cited == [1] and last.invalid_citations == []
    assert len(llm.calls) == 1 and "Question: what does reno do after a timeout" in llm.calls[0][1]["content"]


def test_max_context_limits_what_the_llm_sees():
    llm = FakeLLM()
    ans = RAGAnswerer(StubRetriever(CHUNKS), llm, max_context=1).answer("congestion window", k=3)
    assert len(ans.hits) == 2 and len(ans.context) == 1          # 2 matching chunks shown, 1 sent
    assert "[2]" not in llm.calls[0][1]["content"]


def test_no_hits_skips_llm():
    llm = FakeLLM()
    ans = RAGAnswerer(StubRetriever(CHUNKS), llm).answer("quantum entanglement")
    assert ans.text == NO_HITS and ans.done and not ans.llm_used and llm.calls == []


def test_low_confidence_skips_llm_only_when_asked():
    low = confidence_model(0.1)
    llm = FakeLLM()
    ans = RAGAnswerer(StubRetriever(CHUNKS), llm, low, skip_when_low=True).answer("congestion window")
    assert ans.text == LOW_CONFIDENCE and ans.confidence.level == "low" and llm.calls == []
    assert ans.hits  # sources still shown

    ans = RAGAnswerer(StubRetriever(CHUNKS), llm, low).answer("congestion window")
    assert ans.llm_used and ans.confidence.level == "low"


def test_confidence_uses_its_own_k_even_when_fewer_sources_are_shown():
    conf = confidence_model(0.9)  # k=3
    retriever = StubRetriever(CHUNKS)
    seen = []
    original = retriever.search
    retriever.search = lambda q, k=5, source=None: seen.append(k) or original(q, k, source)
    ans = RAGAnswerer(retriever, FakeLLM(), conf).answer("congestion window", k=1)
    assert seen == [3] and len(ans.hits) == 1 and ans.confidence.level == "high"


# ---------------------------------------------------------------------------------------- UI

def ui_fn(app, name):
    return next(f.fn for f in app.fns.values() if getattr(f.fn, "__name__", "") == name)


def test_ui_streams_answer_and_marks_cited_sources():
    from Interface_03.app import ALL_SOURCES, build_app

    retriever = StubRetriever(CHUNKS)
    app = build_app(retriever, "stub", answerer=RAGAnswerer(retriever, FakeLLM()))
    updates = list(ui_fn(app, "ask")("what does reno do after a timeout", 2, ALL_SOURCES))

    assert updates[0][0] == "_Reading the sources…_"
    assert updates[1][0].endswith("▌")                       # still streaming
    answer, sources, status = updates[-1]
    assert answer == "TCP Reno resets cwnd to 1 MSS [1]."
    assert sources.count('class="hit hit-cited"') == 1 and "cited" in sources
    assert "fake-llm" in status


def test_ui_warns_about_uncited_or_invented_citations():
    from Interface_03.app import ALL_SOURCES, build_app

    retriever = StubRetriever(CHUNKS)
    for pieces, warning in [(("No sources used.",), "cites no source"), (("See [9].",), "don't exist: [9]")]:
        app = build_app(retriever, "stub", answerer=RAGAnswerer(retriever, FakeLLM(pieces)))
        assert warning in list(ui_fn(app, "ask")("congestion window", 2, ALL_SOURCES))[-1][0]


def test_ui_survives_llm_errors_and_keeps_sources():
    from Interface_03.app import ALL_SOURCES, build_app

    retriever = StubRetriever(CHUNKS)
    llm = FakeLLM(pieces=(), error=ConnectionError("connection refused"))
    app = build_app(retriever, "stub", answerer=RAGAnswerer(retriever, llm))
    answer, sources, _ = list(ui_fn(app, "ask")("congestion window", 2, ALL_SOURCES))[-1]
    assert "Couldn't get an answer" in answer and "ollama serve" in answer
    assert 'class="hit"' in sources


def test_ui_without_llm_is_search_only():
    from Interface_03.app import build_app

    app = build_app(StubRetriever(CHUNKS), "stub")
    names = {getattr(f.fn, "__name__", "") for f in app.fns.values()}
    assert "search" in names and "ask" not in names
