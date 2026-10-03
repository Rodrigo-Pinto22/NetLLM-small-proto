import sys
import types

from Ingestion_01.doc_dataclass import Page
from Ingestion_01.parsers import PdfParser, TextParser


def test_text_parser_reads_whole_file_as_one_page(tmp_path):
    f = tmp_path / "rfc793.txt"
    f.write_text("# TCP\nTransmission Control Protocol\n", encoding="utf-8")
    doc = TextParser().parse(f)
    assert doc.source == "rfc793.txt"
    assert doc.pages == [Page("# TCP\nTransmission Control Protocol\n", 1)]
    assert doc.metadata == {}


def test_text_parser_ignores_invalid_utf8(tmp_path):
    f = tmp_path / "bad.md"
    f.write_bytes(b"ok \xff\xfe text")
    assert TextParser().parse(f).pages[0].text == "ok  text"


def test_pdf_parser_maps_page_chunks(tmp_path, monkeypatch):
    calls = []

    def to_markdown(path, page_chunks):
        calls.append((path, page_chunks))
        return [
            {"text": "# Ch 1\nfirst", "metadata": {"page": 1}},
            {"text": "second", "metadata": {"page": 2}},
            {"text": "no metadata"},  # falls back to the enumeration index
        ]

    monkeypatch.setitem(sys.modules, "pymupdf4llm", types.SimpleNamespace(to_markdown=to_markdown))
    pdf = tmp_path / "book.pdf"
    doc = PdfParser().parse(pdf)

    assert calls == [(str(pdf), True)]
    assert doc.source == "book.pdf"
    assert doc.pages == [Page("# Ch 1\nfirst", 1), Page("second", 2), Page("no metadata", 3)]
