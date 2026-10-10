import json
import random

import pytest

from Encoder_02.retriever import Hit
from Evaluation_04.baselines import BM25Retriever
from Evaluation_04.confidence import (
    ConfidenceModel, IsotonicCalibrator, auroc, brier, ece, fit_confidence, signals,
)
from Evaluation_04.dataset import EvalItem, group_split, load_eval_set
from Evaluation_04.evaluate import QueryResult, compare, judge, metric_values, summarize, summarize_by_type
from Evaluation_04.matching import is_relevant
from Evaluation_04.metrics import (
    bootstrap_ci, first_hit_rank, paired_bootstrap, recall_at_k, reciprocal_rank,
)
from Ingestion_01.doc_dataclass import Chunk

TCP = "3 Transport > 3.7 Congestion Control"


def chunk(id, text, section=TCP, source="book.pdf"):
    return Chunk(id, text, {"source": source, "section": section, "page": 1, "part": 0})


def item(query, evidence, section=TCP, type="factual"):
    return EvalItem(query, type, evidence, [{"source": "book.pdf", "section": section, "page": 1}])


CHUNKS = [
    chunk("reno", "After a timeout, TCP Reno sets the congestion window to one segment and restarts slow start."),
    chunk("fast", "Fast recovery halves the congestion window after three duplicate acknowledgements.",
          section=TCP + " > Fast Recovery"),
    chunk("ospf", "OSPF floods link-state advertisements to all routers in the area.",
          section="4 Network > 4.6.2 OSPF"),
    chunk("switch", "Switches are plug-and-play devices that learn MAC addresses by themselves.",
          section="5 Link > 5.4.3 Switches"),
]
ITEMS = [
    item("What does TCP Reno do to cwnd on a timeout?",
         "After a timeout, TCP Reno sets the congestion window to one segment and restarts slow start."),
    item("How are link-state advertisements distributed in OSPF?",
         "OSPF floods link-state advertisements to all routers in the area.", section="4 Network > 4.6.2 OSPF",
         type="keyword"),
    item("Why don't switches need configuration?",
         "Switches are plug-and-play devices that learn MAC addresses by themselves.",
         section="5 Link > 5.4.3 Switches", type="conceptual"),
]


# ---------------------------------------------------------------------------------- dataset

def test_load_eval_set_reads_generator_output(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"query": "q?", "type": "keyword", "evidence": "e",
                             "relevant": [{"source": "b.pdf", "section": "S", "page": 3}]}) + "\n\n")
    [it] = load_eval_set(p)
    assert it == EvalItem("q?", "keyword", "e", [{"source": "b.pdf", "section": "S", "page": 3}])
    assert it.section_key == "b.pdf|S"


def test_group_split_keeps_groups_together_and_is_deterministic():
    keys = [f"s{i % 10}" for i in range(100)]
    train, test = group_split(keys, test_frac=0.3, seed=1)
    assert sorted(train + test) == list(range(100))
    assert not {keys[i] for i in train} & {keys[i] for i in test}
    assert len({keys[i] for i in test}) == 3
    assert (train, test) == group_split(keys, test_frac=0.3, seed=1)


# --------------------------------------------------------------------------------- matching

def test_relevance_modes():
    it = ITEMS[0]
    assert is_relevant(CHUNKS[0], it, "evidence")
    assert not is_relevant(CHUNKS[1], it, "evidence")
    assert is_relevant(CHUNKS[0], it, "section")
    assert is_relevant(CHUNKS[1], it, "section")  # sub-section of the labelled one
    assert not is_relevant(CHUNKS[2], it, "section")
    assert not is_relevant(chunk("x", "...", section=TCP + "X"), it, "section")  # prefix of a sibling
    assert not is_relevant(chunk("y", "...", source="other.pdf"), it, "section")
    with pytest.raises(ValueError):
        is_relevant(CHUNKS[0], it, "vibes")


def test_evidence_relevance_tolerates_chunk_prefix_and_unicode():
    c = chunk("p", "3 Transport > 3.7 Congestion Control\n\nAfter a time‑out, TCP **Reno** sets the "
                   "congestion window to one segment and restarts slow start.")
    assert is_relevant(c, ITEMS[0], "evidence")


# ---------------------------------------------------------------------------------- metrics

def test_rank_metrics():
    hits = [False, False, True, False, True]
    assert recall_at_k(hits, 2) == 0.0 and recall_at_k(hits, 3) == 1.0
    assert reciprocal_rank(hits, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank(hits, 2) == 0.0
    assert first_hit_rank(hits) == 3 and first_hit_rank([False]) is None
    assert recall_at_k([], 5) == 0.0


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 0, 1, 1, 1] * 10)
    assert mean == pytest.approx(0.7)
    assert lo < 0.7 < hi and hi - lo < 0.25
    assert bootstrap_ci([1.0] * 5) == (1.0, 1.0, 1.0)


def test_paired_bootstrap_detects_real_difference_only():
    rng = random.Random(0)
    b = [float(rng.random() < 0.5) for _ in range(300)]
    better = [1.0 if rng.random() < 0.3 else x for x in b]   # fixes ~30% of B's misses
    assert paired_bootstrap(better, b)["p"] < 0.05
    same = paired_bootstrap(b, b)
    assert same["diff"] == 0 and same["p"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


# ------------------------------------------------------------------------------------- BM25

def test_bm25_ranks_matching_chunk_first_and_filters():
    bm25 = BM25Retriever(CHUNKS + [chunk("udp", "UDP congestion", source="paper.pdf")])
    hits = bm25.search("OSPF link-state advertisements", k=2)
    assert hits[0].chunk.id == "ospf" and hits[0].score > (hits[1].score if len(hits) > 1 else 0)
    assert [h.chunk.id for h in bm25.search("congestion", source="paper.pdf")] == ["udp"]
    assert bm25.search("quantum") == []
    assert bm25.sources() == ["book.pdf", "paper.pdf"]


def test_bm25_rare_terms_outweigh_common_ones():
    bm25 = BM25Retriever(CHUNKS)
    # "the" is everywhere-ish, "MAC" only in one chunk
    assert bm25.search("the MAC", k=1)[0].chunk.id == "switch"


# ---------------------------------------------------------------------------------- evaluate

@pytest.fixture
def results():
    return judge(BM25Retriever(CHUNKS), ITEMS, depth=3)


def test_judge_marks_relevance_per_mode(results):
    assert len(results) == 3
    r = results[0]
    assert r.retrieved[0]["id"] == "reno"
    assert r.hits["evidence"][0] and r.hits["section"][0]
    assert r.scores == sorted(r.scores, reverse=True)
    assert r.group == f"book.pdf|{TCP}"


def test_summaries(results):
    s = summarize(results, "evidence", ["recall@1", "mrr@10"], n_boot=200)
    assert s["recall@1"][0] == 1.0 and s["mrr@10"][0] == 1.0
    by_type = summarize_by_type(results, "evidence", "recall@1", n_boot=200)
    assert set(by_type) == {"factual", "keyword", "conceptual"} and by_type["keyword"][0] == 1
    assert metric_values(results, "section", "recall@3") == [1.0, 1.0, 1.0]


def test_compare_requires_same_questions(results):
    assert compare(results, results, "evidence", "recall@1", n_boot=200)["diff"] == 0
    with pytest.raises(ValueError):
        compare(results, results[:2], "evidence", "recall@1")


# ------------------------------------------------------------------------------- confidence

def test_signals():
    assert signals([0.9, 0.5, 0.4]) == pytest.approx({"top1": 0.9, "margin": 0.4, "spread": 0.9 - 0.6})
    assert signals([0.3, 0.8]) ["top1"] == 0.8  # sorts defensively
    assert signals([0.7]) == {"top1": 0.7, "margin": 0.7, "spread": 0.0}
    assert signals([]) == {"top1": 0.0, "margin": 0.0, "spread": 0.0}


def test_auroc_brier_ece():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert auroc([0.5], [True]) != auroc([0.5], [True])  # nan: undefined with one class
    assert brier([1, 0], [True, False]) == 0 and brier([0.5, 0.5], [True, False]) == 0.25
    assert ece([0.9] * 10, [True] * 9 + [False]) == pytest.approx(0.0)


def test_isotonic_is_monotone_smoothed_and_handles_ties():
    x = [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    y = [0, 1, 0, 0, 1, 1, 0, 1]
    cal = IsotonicCalibrator.fit(x, y)
    preds = [cal.predict(v) for v in [0.0, 0.1, 0.2, 0.25, 0.35, 0.5, 0.65, 0.9]]
    assert preds == sorted(preds)
    assert 0 < min(preds) and max(preds) < 1           # Laplace smoothing: never 0% / 100%
    assert len(set(cal.xs)) == len(cal.xs)             # one step per distinct value
    with pytest.raises(ValueError):
        IsotonicCalibrator.fit([], [])


def synthetic_results(n=300, seed=0):
    """top1 score predicts success; margin and spread are noise."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        top1 = rng.random()
        ok = rng.random() < top1
        scores = [top1, top1 - rng.random() * 0.05, top1 - 0.1]
        out.append(QueryResult(f"q{i}", "factual", f"sec{i % 60}", scores,
                               {"evidence": [ok, False, False], "section": [ok, False, False]}, []))
    return out


def test_fit_confidence_picks_predictive_signal_and_is_calibrated():
    model = fit_confidence(synthetic_results(), "bm25", k=3)
    assert model.signal == "top1"
    assert model.report["test_auroc"] > 0.7 and model.report["test_ece"] < 0.15
    assert model.report["n_train"] + model.report["n_test"] == 300
    low = model.assess([Hit(CHUNKS[0], 0.05), Hit(CHUNKS[1], 0.01)])
    high = model.assess([Hit(CHUNKS[0], 0.97), Hit(CHUNKS[1], 0.90)])
    assert low.probability < 0.3 < 0.7 < high.probability
    assert (low.level, high.level) == ("low", "high")


def test_fit_confidence_needs_several_sections():
    one_section = [QueryResult("q", "t", "same", [1.0], {"evidence": [True]}, [])] * 10
    with pytest.raises(ValueError):
        fit_confidence(one_section, "bm25")


def test_confidence_model_save_load_roundtrip(tmp_path):
    model = fit_confidence(synthetic_results(), "bm25", k=3)
    model.save(tmp_path / "sub" / "c.json")
    loaded = ConfidenceModel.load(tmp_path / "sub" / "c.json")
    assert loaded == model
    hits = [Hit(CHUNKS[0], 0.6)]
    assert loaded.assess(hits) == model.assess(hits)


# ------------------------------------------------------------------------------------- CLI

def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    from Evaluation_04 import run
    from Ingestion_01.ingestor import write_jsonl

    write_jsonl(CHUNKS, tmp_path / "chunks.jsonl")
    with open(tmp_path / "q.jsonl", "w") as f:
        for it in ITEMS:
            f.write(json.dumps(it.__dict__) + "\n")
    monkeypatch.setattr("sys.argv", ["run", "--eval", str(tmp_path / "q.jsonl"), "--chunks", str(tmp_path / "chunks.jsonl"),
                                     "--retrievers", "bm25", "stub", "--ks", "1", "3", "--out", str(tmp_path / "res")])
    run.main()
    out = capsys.readouterr().out
    assert "== evidence relevance" in out and "== section relevance" in out
    assert "stub - bm25 (recall@5)" in out
    rows = [json.loads(ln) for ln in open(tmp_path / "res" / "bm25.jsonl")]
    assert rows[0]["first_hit"]["evidence"] == 1


def test_cli_refuses_mismatched_book(tmp_path, monkeypatch):
    from Evaluation_04 import run
    from Ingestion_01.ingestor import write_jsonl

    write_jsonl([chunk("a", "text", source="NetLLM_paper.pdf")], tmp_path / "chunks.jsonl")
    (tmp_path / "q.jsonl").write_text(json.dumps(ITEMS[0].__dict__) + "\n")
    monkeypatch.setattr("sys.argv", ["run", "--eval", str(tmp_path / "q.jsonl"), "--chunks", str(tmp_path / "chunks.jsonl")])
    with pytest.raises(SystemExit, match="Ingest the same book first"):
        run.main()


# -------------------------------------------------------------------------------------- UI

def test_ui_shows_calibrated_confidence_using_its_own_k():
    from Interface_03.app import build_app

    model = fit_confidence(synthetic_results(), "bm25", k=3)
    app = build_app(BM25Retriever(CHUNKS), "bm25", model)
    search = next(f.fn for f in app.fns.values() if getattr(f.fn, "__name__", "") == "search")
    html, status = search("TCP Reno timeout congestion window", 1, "All documents")
    assert html.count('class="hit"') == 1             # shows k=1 ...
    assert "Retrieval confidence" in status and "in the top 3" in status  # ... but judges top 3


# ----------------------------------------------------------------- cluster bootstrap + report

def test_cluster_bootstrap_widens_ci_for_correlated_groups():
    # 40 sections x 10 questions; within a section all questions succeed or fail together.
    rng = random.Random(1)
    values, groups = [], []
    for g in range(40):
        outcome = float(rng.random() < 0.6)
        values += [outcome] * 10
        groups += [f"s{g}"] * 10
    _, qlo, qhi = bootstrap_ci(values)
    _, slo, shi = bootstrap_ci(values, groups=groups)
    assert (shi - slo) > 2 * (qhi - qlo)   # ~sqrt(10)x wider: only 40 independent outcomes


def test_cluster_bootstrap_falls_back_with_one_group_and_validates_length():
    v = [1, 0, 1, 1, 0, 1]
    assert bootstrap_ci(v, groups=["same"] * 6) == bootstrap_ci(v)
    with pytest.raises(ValueError):
        bootstrap_ci(v, groups=["a"])
    with pytest.raises(ValueError):
        paired_bootstrap(v, v, groups=["a", "b"])


def test_paired_bootstrap_by_group_still_detects_real_difference():
    rng = random.Random(0)
    groups = [f"s{i // 4}" for i in range(400)]
    b = [float(rng.random() < 0.5) for _ in groups]
    better = [1.0 if rng.random() < 0.3 else x for x in b]
    assert paired_bootstrap(better, b, groups=groups)["p"] < 0.05


def test_summaries_resample_by_section_by_default(results):
    # Same questions duplicated within their sections: by-section CI must be wider.
    many = [r for r in results for _ in range(10)]
    by_q = summarize(many, "section", ["recall@1"], n_boot=500, by_section=False)["recall@1"]
    by_s = summarize(many, "section", ["recall@1"], n_boot=500)["recall@1"]
    assert by_q[0] == by_s[0]


def test_write_and_load_results_roundtrip(results, tmp_path):
    from Evaluation_04.evaluate import load_results, write_results

    write_results(results, tmp_path / "r" / "bm25.jsonl")
    assert load_results(tmp_path / "r" / "bm25.jsonl") == results


def test_report_cli_from_saved_results(tmp_path, monkeypatch, capsys):
    from Evaluation_04 import report, run
    from Ingestion_01.ingestor import write_jsonl

    write_jsonl(CHUNKS, tmp_path / "chunks.jsonl")
    with open(tmp_path / "q.jsonl", "w") as f:
        for it in ITEMS:
            f.write(json.dumps(it.__dict__) + "\n")
    monkeypatch.setattr("sys.argv", ["run", "--eval", str(tmp_path / "q.jsonl"), "--chunks", str(tmp_path / "chunks.jsonl"),
                                     "--retrievers", "bm25", "stub", "--ks", "1", "3", "--out", str(tmp_path / "res")])
    run.main()
    capsys.readouterr()

    monkeypatch.setattr("sys.argv", ["report", str(tmp_path / "res" / "stub.jsonl"), str(tmp_path / "res" / "bm25.jsonl"),
                                     "--ks", "1", "3"])
    report.main()
    out = capsys.readouterr().out
    assert "3 questions, retrievers: stub, bm25" in out
    assert "95% CI by section, 3 sections" in out
    assert "bm25 - stub (recall@5)" in out

    monkeypatch.setattr("sys.argv", ["report", str(tmp_path / "res"), "--ks", "1", "3", "--ci", "question"])
    report.main()
    assert "95% CI by question" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["report", str(tmp_path / "res")])   # default ks include 20 > depth 10
    with pytest.raises(SystemExit, match="depth 10"):
        report.main()
