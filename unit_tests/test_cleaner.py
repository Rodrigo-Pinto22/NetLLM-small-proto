from Ingestion_01.cleaner import Cleaner
from Ingestion_01.doc_dataclass import Document, Page


def letters(i: int) -> str:
    return "".join(chr(ord("a") + int(d)) for d in str(i))


def make_doc(texts: list[str], source: str = "book.pdf") -> Document:
    return Document(source, [Page(t, i) for i, t in enumerate(texts, 1)], {"k": "v"})


def test_rejoins_hyphenated_words():
    out = Cleaner().clean(make_doc(["The connec-\ntion is reliable."]))
    assert out.pages[0].text == "The connection is reliable."


def test_drops_bare_page_numbers():
    out = Cleaner().clean(make_doc(["Some body text.\n\n215"]))
    assert out.pages[0].text == "Some body text."


def test_keeps_numbers_inside_text():
    out = Cleaner().clean(make_doc(["Port 443 is used by HTTPS.\n12345"]))
    assert out.pages[0].text == "Port 443 is used by HTTPS.\n12345"


def test_collapses_whitespace_and_blank_lines():
    out = Cleaner().clean(make_doc(["  a\t\t b   c  \n\n\n\n\nnext"]))
    assert out.pages[0].text == "a b c\n\nnext"


def test_removes_repeated_headers_and_footers_ignoring_digits():
    texts = [
        f"Chapter 3 • Transport Layer {200 + i}\nBody of page {i}.\nMore text here.\nComputer Networking"
        for i in range(6)
    ]
    out = Cleaner().clean(make_doc(texts))
    for i, page in enumerate(out.pages):
        assert page.text == f"Body of page {i}.\nMore text here."


def test_no_boilerplate_detection_below_min_repeats():
    texts = ["Running Header\nBody."] * 4
    out = Cleaner(min_repeats=5).clean(make_doc(texts))
    assert all(p.text == "Running Header\nBody." for p in out.pages)


def test_threshold_scales_with_page_count():
    # 200 pages * 0.03 = 6 repeats needed; a line seen on 5 page edges is not boilerplate.
    bodies = [f"Body {letters(i)}." for i in range(200)]  # unique even after digits are ignored
    texts = [f"Rare Line\n{b}" if i < 5 else b for i, b in enumerate(bodies)]
    out = Cleaner(min_repeats=5, repeat_ratio=0.03).clean(make_doc(texts))
    assert out.pages[0].text == f"Rare Line\n{bodies[0]}"


def test_only_edge_lines_count_towards_boilerplate():
    # "Middle" repeats on every page but is never in the first/last two lines.
    texts = [f"T{i}\nU{i}x\nMiddle\nV{i}y\nW{i}z" for i in range(6)]
    out = Cleaner().clean(make_doc(texts))
    assert all("Middle" in p.text for p in out.pages)


def test_drops_pages_that_become_empty_and_keeps_numbers_and_metadata():
    out = Cleaner().clean(make_doc(["Real text.", "  \n 42 \n", "More text."]))
    assert [p.number for p in out.pages] == [1, 3]
    assert out.source == "book.pdf"
    assert out.metadata == {"k": "v"}


def test_does_not_mutate_input():
    doc = make_doc(["a-\nb  c"])
    Cleaner().clean(doc)
    assert doc.pages[0].text == "a-\nb  c"
