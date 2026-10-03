from Ingestion_01.doc_dataclass import Document, Page, Section
from Ingestion_01.splitter import SectionSplitter


def split(*texts: str) -> list[Section]:
    doc = Document("doc.md", [Page(t, i) for i, t in enumerate(texts, 1)])
    return SectionSplitter().split(doc)


def test_text_without_headings_is_one_section_with_empty_path():
    assert split("Just some text.\nAnother line.") == [Section([], "Just some text.\nAnother line.", 1)]


def test_one_section_per_heading():
    sections = split("# Intro\nHello.\n# Body\nWorld.")
    assert [(s.path, s.text) for s in sections] == [(["Intro"], "Hello."), (["Body"], "World.")]


def test_nested_heading_path():
    sections = split("# Transport Layer\nA.\n## 3.5 TCP\nB.\n### Handshake\nC.\n## 3.4 UDP\nD.")
    assert [s.path for s in sections] == [
        ["Transport Layer"],
        ["Transport Layer", "3.5 TCP"],
        ["Transport Layer", "3.5 TCP", "Handshake"],
        ["Transport Layer", "3.4 UDP"],
    ]


def test_skipped_heading_level_does_not_crash():
    sections = split("# A\nx\n### C\ny")
    assert [s.path for s in sections] == [["A"], ["A", "C"]]


def test_strips_markdown_emphasis_from_titles():
    assert split("## **Bold Title**\ntext")[0].path == ["Bold Title"]


def test_heading_without_body_produces_no_section():
    sections = split("# Empty\n# Full\ncontent")
    assert [s.path for s in sections] == [["Full"]]


def test_hash_without_space_is_not_a_heading():
    sections = split("#hashtag line\nmore")
    assert sections == [Section([], "#hashtag line\nmore", 1)]


def test_section_page_is_where_its_heading_starts():
    sections = split("# One\nfirst page", "continues here\n# Two\nsecond", "third")
    assert [(s.path, s.page) for s in sections] == [(["One"], 1), (["Two"], 2)]
    assert sections[0].text == "first page\ncontinues here"
    assert sections[1].text == "second\nthird"


def test_leading_text_page_is_first_non_blank_line():
    sections = split("", "\n\nintro text")
    assert sections == [Section([], "intro text", 2)]


def test_section_paths_are_independent_copies():
    sections = split("# A\nx\n## B\ny")
    sections[0].path.append("mutated")
    assert sections[1].path == ["A", "B"]
