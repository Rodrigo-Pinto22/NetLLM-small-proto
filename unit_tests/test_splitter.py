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


def test_same_level_headings_without_parent_are_siblings():
    # Regression: "##" headings with no "#" above used to nest under the first one.
    sections = split("## ABSTRACT\na\n## CCS CONCEPTS\nb\n## KEYWORDS\nc")
    assert [s.path for s in sections] == [["ABSTRACT"], ["CCS CONCEPTS"], ["KEYWORDS"]]


def test_numbering_defines_hierarchy_when_markdown_levels_are_flat():
    sections = split("## 4 NETLLM DESIGN\na\n## 4.1 Multimodal Encoder\nb\n## 4.2 Networking Head\nc\n"
                     "## 5 EVALUATION\nd\n## 5.1 Setup\ne\n### 5.1.1 Hardware\nf")
    assert [s.path for s in sections] == [
        ["4 NETLLM DESIGN"],
        ["4 NETLLM DESIGN", "4.1 Multimodal Encoder"],
        ["4 NETLLM DESIGN", "4.2 Networking Head"],
        ["5 EVALUATION"],
        ["5 EVALUATION", "5.1 Setup"],
        ["5 EVALUATION", "5.1 Setup", "5.1.1 Hardware"],
    ]


def test_netllm_paper_layout():
    # Heading levels exactly as pymupdf4llm emits them for the NetLLM paper (title removed by Cleaner).
    headings = ["## ABSTRACT", "#### ACM Reference Format:", "## 1 INTRODUCTION", "## 1.1 The Main Roadmap",
                "## 6 DISCUSSION", "### **_Q2: How does_ NetLLM** **_compare to RAG?_**",
                "## 7 CONCLUDING REMARKS", "## REFERENCES", "## A APPENDICES", "## A.1 Details of Figure 2"]
    sections = split("\n".join(f"{h}\ntext" for h in headings))
    assert [s.path for s in sections] == [
        ["ABSTRACT"],
        ["ABSTRACT", "ACM Reference Format:"],
        ["1 INTRODUCTION"],
        ["1 INTRODUCTION", "1.1 The Main Roadmap"],
        ["6 DISCUSSION"],
        ["6 DISCUSSION", "Q2: How does NetLLM compare to RAG?"],
        ["7 CONCLUDING REMARKS"],
        ["REFERENCES"],
        ["A APPENDICES"],
        ["A APPENDICES", "A.1 Details of Figure 2"],
    ]


def test_unnumbered_document_keeps_markdown_levels():
    sections = split("# Guide\na\n## Install\nb\n### Linux\nc\n## Usage\nd")
    assert [s.path for s in sections] == [["Guide"], ["Guide", "Install"], ["Guide", "Install", "Linux"], ["Guide", "Usage"]]


def test_years_are_not_section_numbers():
    sections = split("# Report\na\n## 2019 Results\nb")
    assert [s.path for s in sections] == [["Report"], ["Report", "2019 Results"]]
