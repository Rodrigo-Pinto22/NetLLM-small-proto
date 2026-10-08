import json

import pytest

from Encoder_02.datasetCreation import (
    PROMPT, evidence_ok, generate, load_sections, norm, read_jsonl, section_key, select_sections,
)
from Ingestion_01.doc_dataclass import Section

TEXT = ("TCP uses a congestion window to limit the sending rate. "
        "After a timeout, TCP Reno sets the congestion window to one segment. "
        "The receive window field advertises free buffer space.")


def sec(path, text=TEXT, page=1):
    return Section(path, text, page)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def q(query, evidence="After a timeout, TCP Reno sets the congestion window to one segment.", type="factual"):
    return {"query": query, "type": type, "evidence": evidence}


def run(sections, llm, out, limit=10, **kw):
    return generate(sections, llm, "book.pdf", out, limit, log=lambda _: None, **kw)


# ----------------------------------------------------------------- normalisation / evidence

def test_norm_unifies_dashes_quotes_markdown_and_whitespace():
    assert norm("**Data‑driven**  “RL”\n it’s") == 'data-driven "rl" it\'s'


def test_evidence_exact_and_near_matches_pass():
    assert evidence_ok("After a timeout, TCP Reno sets the congestion window to one segment.", TEXT)
    assert evidence_ok("after a time-out, TCP Reno sets the congestion window to one segment", TEXT)


def test_evidence_not_in_section_fails():
    assert not evidence_ok("UDP provides no reliability guarantees whatsoever to applications.", TEXT)


def test_evidence_too_short_fails():
    assert not evidence_ok("TCP", TEXT)


# ----------------------------------------------------------------------- section selection

def test_select_filters_length_and_non_content_headings():
    long_text = "word " * 200
    sections = [
        sec(["3 Transport", "3.5 TCP"], long_text),
        sec(["3 Transport", "Homework Problems and Questions"], long_text),
        sec(["References"], long_text),
        sec(["3 Transport", "3.8 Summary"], long_text),
        sec(["1 Intro"], "too short"),
        sec(["2 Huge"], "word " * 5000),
    ]
    picked = select_sections(sections, min_words=120, max_words=3500, seed=0)
    assert [s.path for s in picked] == [["3 Transport", "3.5 TCP"]]


def test_select_order_is_deterministic_and_limits_nest():
    sections = [sec([f"{i} S"], "word " * 200, i) for i in range(30)]
    a = select_sections(sections, 120, 3500, seed=0)
    b = select_sections(sections, 120, 3500, seed=0)
    assert a == b
    assert a[:5] == b[:20][:5]  # a pilot of 5 is the start of a run of 20
    assert a != select_sections(sections, 120, 3500, seed=1)


# -------------------------------------------------------------------------------- generate

def test_generate_keeps_good_questions_with_labels(tmp_path):
    out = tmp_path / "eval.jsonl"
    llm = FakeLLM([q("What does TCP Reno do to cwnd after a timeout?")])
    stats = run([sec(["3 Transport", "3.7 Congestion"], page=42)], llm, out)

    assert stats["kept"] == 1
    assert read_jsonl(out) == [{
        "query": "What does TCP Reno do to cwnd after a timeout?",
        "type": "factual",
        "evidence": "After a timeout, TCP Reno sets the congestion window to one segment.",
        "relevant": [{"source": "book.pdf", "section": "3 Transport > 3.7 Congestion", "page": 42}],
    }]
    assert "<section_path>3 Transport > 3.7 Congestion</section_path>" in llm.prompts[0]
    assert TEXT in llm.prompts[0]


def test_generate_rejects_bad_questions_with_reason(tmp_path):
    out = tmp_path / "eval.jsonl"
    llm = FakeLLM([
        q("Good question?"),
        q("Hallucinated evidence?", evidence="QUIC runs over UDP and integrates TLS 1.3 handshakes."),
        q("good   QUESTION?"),                 # duplicate after normalisation
        q("", ),                               # malformed: empty
        q("Bad type?", type="opinion"),        # malformed: unknown type
    ])
    stats = run([sec(["S"])], llm, out)

    assert stats["kept"] == 1 and stats["rejected"] == 4
    reasons = [r["reason"] for r in read_jsonl(out.with_suffix(".rejected.jsonl"))]
    assert reasons == ["evidence not in section", "duplicate", "malformed", "malformed"]


def test_generate_resumes_and_extends(tmp_path):
    out = tmp_path / "eval.jsonl"
    sections = [sec([f"S{i}"], page=i) for i in range(4)]

    run(sections, FakeLLM([q("Q0?")], [q("Q1?")]), out, limit=2)
    llm = FakeLLM([q("Q2?")], [q("Q3?")])
    stats = run(sections, llm, out, limit=4)

    assert stats["already_done"] == 2 and stats["sections"] == 2
    assert [r["query"] for r in read_jsonl(out)] == ["Q0?", "Q1?", "Q2?", "Q3?"]
    assert all("<section_path>S0" not in p and "<section_path>S1" not in p for p in llm.prompts)


def test_generate_dedups_against_previous_runs(tmp_path):
    out = tmp_path / "eval.jsonl"
    run([sec(["A"], page=1)], FakeLLM([q("Same question?")]), out)
    stats = run([sec(["A"], page=1), sec(["B"], page=2)], FakeLLM([q("same question?")]), out)
    assert stats["kept"] == 0 and stats["rejected"] == 1


def test_empty_answer_marks_section_done(tmp_path):
    out = tmp_path / "eval.jsonl"
    run([sec(["References-like"])], FakeLLM([]), out)
    stats = run([sec(["References-like"])], FakeLLM(), out)  # would raise if asked again
    assert stats["already_done"] == 1 and stats["sections"] == 0


def test_llm_failure_retries_then_leaves_section_for_next_run(tmp_path):
    out = tmp_path / "eval.jsonl"
    stats = run([sec(["S"])], FakeLLM(ValueError("bad json"), ValueError("bad json")), out)
    assert stats["failed"] == 1 and stats["sections"] == 0
    assert not out.with_suffix(".progress.jsonl").exists()

    stats = run([sec(["S"])], FakeLLM(ValueError("hiccup"), [q("Recovered?")]), out)
    assert stats["kept"] == 1


def test_section_key_distinguishes_repeated_headings():
    assert section_key("b.pdf", sec(["Summary"], page=10)) != section_key("b.pdf", sec(["Summary"], page=90))


# ---------------------------------------------------------------------------------- loading

def test_load_sections_uses_cache_until_book_changes(tmp_path):
    book = tmp_path / "book.md"
    book.write_text("# 1 Intro\nHello there.", encoding="utf-8")
    cache = tmp_path / "data" / "book.sections.json"

    assert load_sections(book, cache) == [Section(["1 Intro"], "Hello there.", 1)]
    cache.write_text(json.dumps([{"path": ["cached"], "text": "x", "page": 1}]), encoding="utf-8")
    assert load_sections(book, cache)[0].path == ["cached"]  # cache hit

    import os
    os.utime(book, (cache.stat().st_mtime + 10,) * 2)  # book edited after the cache
    assert load_sections(book, cache)[0].path == ["1 Intro"]


def test_prompt_placeholders_and_object_output():
    assert "{section_path}" in PROMPT and "{section_text}" in PROMPT
    assert '{"questions": [' in PROMPT
