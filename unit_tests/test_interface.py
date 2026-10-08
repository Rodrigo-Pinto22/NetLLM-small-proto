import json

import pytest

from Encoder_02.retriever import Hit, StubRetriever, load_chunks
from Ingestion_01.doc_dataclass import Chunk


def chunk(id, text, source="book.pdf", section="3 Transport > 3.5 TCP", page=1, part=0):
    return Chunk(id, text, {"source": source, "section": section, "page": page, "part": part})


@pytest.fixture
def retriever():
    return StubRetriever([
        chunk("a", "TCP Reno halves the congestion window on triple duplicate ACKs."),
        chunk("b", "After a timeout TCP sets the congestion window to one segment."),
        chunk("c", "UDP has no congestion control.", source="paper.pdf"),
        chunk("d", "Switches are plug-and-play devices."),
    ])


# ------------------------------------------------------------------------------ retriever

def test_stub_ranks_by_query_word_overlap(retriever):
    hits = retriever.search("what happens to the congestion window after a timeout?", k=5)
    assert [h.chunk.id for h in hits] == ["b", "a", "c"]
    assert hits[0].score > hits[1].score


def test_stub_respects_k_and_source(retriever):
    assert len(retriever.search("congestion", k=1)) == 1
    assert [h.chunk.id for h in retriever.search("congestion", source="paper.pdf")] == ["c"]


def test_stub_no_match_and_empty_query(retriever):
    assert retriever.search("quantum entanglement") == []
    assert retriever.search("  ?! ") == []


def test_sources(retriever):
    assert retriever.sources() == ["book.pdf", "paper.pdf"]


def test_load_chunks_reads_ingestor_output(tmp_path):
    from Ingestion_01.ingestor import write_jsonl

    chunks = [chunk("x", "héllo"), chunk("y", "world")]
    write_jsonl(chunks, tmp_path / "chunks.jsonl")
    assert load_chunks(tmp_path / "chunks.jsonl") == chunks


# ------------------------------------------------------------------------------------- UI

def test_render_hits_shows_metadata_and_escapes_html():
    from Interface_03.app import render_hits

    out = render_hits([Hit(chunk("id1", "if a < b && <script>x</script>", page=42, part=1), 0.5)])
    assert "[1]" in out and "0.500" in out
    assert "3 Transport &gt; 3.5 TCP" in out
    assert "book.pdf · p. 42 · part 1 · id id1" in out
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_render_hits_empty_and_missing_metadata():
    from Interface_03.app import render_hits

    assert "No matching chunks" in render_hits([])
    out = render_hits([Hit(Chunk("z", "text", {}), 1.0)])
    assert "(no heading)" in out and "id z" in out


def test_build_app_wires_search(retriever):
    from Interface_03.app import ALL_SOURCES, build_app

    app = build_app(retriever, name="stub")
    fns = [f.fn for f in app.fns.values()]
    search = next(fn for fn in fns if getattr(fn, "__name__", "") == "search")

    html, status = search("congestion window timeout", 2, ALL_SOURCES)
    assert html.count('class="hit"') == 2 and status.startswith("2 results")
    html, status = search("congestion", 5, "paper.pdf")
    assert html.count('class="hit"') == 1 and status.startswith("1 result ")
    assert search("   ", 5, ALL_SOURCES) == ("", "Type a question and press Enter.")
