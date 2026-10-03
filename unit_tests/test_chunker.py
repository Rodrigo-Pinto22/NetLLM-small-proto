import pytest

from conftest import words
from Ingestion_01.doc_dataclass import Document, Section


@pytest.fixture
def doc():
    return Document("book.pdf", [])


def test_loads_requested_tokenizer(fake_transformers, make_chunker):
    make_chunker(tokenizer_name="some/model")
    assert fake_transformers == ["some/model"]


def test_small_section_becomes_one_chunk_with_heading_prefix(make_chunker, doc):
    chunker = make_chunker(min_tokens=1)
    chunks = chunker.chunk(doc, [Section(["Transport", "TCP"], "TCP is reliable.", 7)])
    assert len(chunks) == 1
    assert chunks[0].text == "Transport > TCP\n\nTCP is reliable."
    assert chunks[0].metadata == {"source": "book.pdf", "section": "Transport > TCP", "page": 7, "part": 0}


def test_no_prefix_when_section_has_no_path(make_chunker, doc):
    chunks = make_chunker(min_tokens=1).chunk(doc, [Section([], "Plain text.", 1)])
    assert chunks[0].text == "Plain text."
    assert chunks[0].metadata["section"] == ""


def test_drops_chunks_below_min_tokens(make_chunker, doc):
    chunks = make_chunker(min_tokens=5).chunk(doc, [Section(["H"], "too short", 1)])
    assert chunks == []


def test_ids_are_deterministic_and_unique(make_chunker, doc):
    sections = [Section(["A"], words(10), 1), Section(["B"], words(10), 2)]
    first = make_chunker(min_tokens=1).chunk(doc, sections)
    second = make_chunker(min_tokens=1).chunk(doc, sections)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == 2
    assert all(len(c.id) == 16 for c in first)


def test_id_depends_on_source(make_chunker):
    chunker = make_chunker(min_tokens=1)
    sec = [Section(["A"], words(10), 1)]
    a = chunker.chunk(Document("a.pdf", []), sec)
    b = chunker.chunk(Document("b.pdf", []), sec)
    assert a[0].id != b[0].id


def test_packs_paragraphs_up_to_budget(make_chunker):
    chunker = make_chunker(max_tokens=10, overlap_tokens=0, min_tokens=1)
    text = "\n\n".join([words(4, "a"), words(4, "b"), words(4, "c")])
    assert chunker._pack(text, 10) == [
        words(4, "a") + "\n\n" + words(4, "b"),
        words(4, "c"),
    ]


def test_overlap_carries_short_last_paragraph(make_chunker):
    chunker = make_chunker(max_tokens=10, overlap_tokens=4, min_tokens=1)
    text = "\n\n".join([words(6, "a"), words(3, "b"), words(6, "c")])
    out = chunker._pack(text, 10)
    assert out == [
        words(6, "a") + "\n\n" + words(3, "b"),
        words(3, "b") + "\n\n" + words(6, "c"),
    ]


def test_no_overlap_when_last_paragraph_is_long(make_chunker):
    chunker = make_chunker(max_tokens=10, overlap_tokens=2, min_tokens=1)
    text = "\n\n".join([words(6, "a"), words(6, "b")])
    assert chunker._pack(text, 10) == [words(6, "a"), words(6, "b")]


def test_long_paragraph_split_on_sentences(make_chunker):
    chunker = make_chunker(min_tokens=1)
    para = " ".join(f"{words(4, f's{i}_')}." for i in range(5))  # five 4-word sentences
    pieces = chunker._split_long(para, 9)
    assert all(chunker.n_tokens(p) <= 9 for p in pieces)
    assert " ".join(pieces) == para
    assert all(p.endswith(".") for p in pieces)


def test_sentence_longer_than_budget_is_hard_cut(make_chunker):
    chunker = make_chunker(min_tokens=1)
    para = words(50)  # no sentence boundaries, like a table row
    pieces = chunker._split_long(para, 10)
    assert len(pieces) > 1
    assert all(chunker.n_tokens(p) <= 10 for p in pieces)
    assert "".join(pieces) == para


def test_every_chunk_fits_max_tokens_plus_overlap(make_chunker, doc):
    max_tokens, overlap = 40, 8
    chunker = make_chunker(max_tokens=max_tokens, overlap_tokens=overlap, min_tokens=1)
    paras = [words(n, f"p{i}_") + "." for i, n in enumerate([5, 30, 7, 60, 3, 25, 12])]
    chunks = chunker.chunk(doc, [Section(["Some", "Heading"], "\n\n".join(paras), 1)])
    assert len(chunks) > 1
    assert [c.metadata["part"] for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert chunker.n_tokens(c.text) <= max_tokens + overlap


def test_no_text_lost(make_chunker, doc):
    chunker = make_chunker(max_tokens=20, overlap_tokens=0, min_tokens=1)
    text = "\n\n".join(words(n, f"p{i}_") + "." for i, n in enumerate([5, 30, 7, 12]))
    chunks = chunker.chunk(doc, [Section([], text, 1)])
    # Hard cuts may split mid-word, so compare with all whitespace removed.
    assert "".join("".join(c.text.split()) for c in chunks) == "".join(text.split())
