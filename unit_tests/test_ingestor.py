import json

import pytest

from conftest import words
from Ingestion_01.doc_dataclass import Chunk, Document, Page


@pytest.fixture
def ingestor_mod(fake_transformers):
    from Ingestion_01 import ingestor

    return ingestor


@pytest.fixture
def ing(ingestor_mod, make_chunker):
    return ingestor_mod.Ingestor(chunker=make_chunker(min_tokens=1))


class RecordingParser:
    def __init__(self):
        self.paths = []

    def parse(self, path):
        self.paths.append(path)
        return Document(path.name, [Page(f"# {path.stem}\n{words(5)}", 1)])


def test_default_stages(ingestor_mod, fake_transformers):
    from Ingestion_01.chunker import Chunker
    from Ingestion_01.cleaner import Cleaner
    from Ingestion_01.splitter import SectionSplitter

    ing = ingestor_mod.Ingestor()
    assert set(ing.parsers) == {".pdf", ".md", ".txt"}
    assert isinstance(ing.cleaner, Cleaner)
    assert isinstance(ing.splitter, SectionSplitter)
    assert isinstance(ing.chunker, Chunker)


def test_load_rejects_unknown_extension(ing, tmp_path):
    with pytest.raises(ValueError, match=r"No parser for '\.docx'"):
        ing.load(tmp_path / "notes.docx")


def test_load_picks_parser_by_case_insensitive_suffix(ingestor_mod, make_chunker, tmp_path):
    parser = RecordingParser()
    ing = ingestor_mod.Ingestor(parsers={".md": parser}, chunker=make_chunker(min_tokens=1))
    doc = ing.load(str(tmp_path / "README.MD"))
    assert parser.paths == [tmp_path / "README.MD"]
    assert doc.source == "README.MD"


def test_ingest_markdown_file_end_to_end(ing, tmp_path):
    f = tmp_path / "tcp.md"
    f.write_text(f"# Transport\n{words(5)}\n\n## TCP\nThe connec-\ntion {words(5)}\n", encoding="utf-8")
    chunks = ing.ingest(f)
    assert [c.metadata["section"] for c in chunks] == ["Transport", "Transport > TCP"]
    assert chunks[1].text.startswith("Transport > TCP\n\nThe connection ")
    assert all(c.metadata["source"] == "tcp.md" for c in chunks)


def test_ingest_text(ing):
    chunks = ing.ingest_text(f"# Pasted\n{words(5)}", source="chat")
    assert len(chunks) == 1
    assert chunks[0].metadata == {"source": "chat", "section": "Pasted", "page": 1, "part": 0}


def test_ingest_dir_recurses_sorted_and_skips_unsupported(ingestor_mod, make_chunker, tmp_path):
    parser = RecordingParser()
    ing = ingestor_mod.Ingestor(parsers={".md": parser}, chunker=make_chunker(min_tokens=1))
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.md").touch()
    (tmp_path / "sub" / "a.md").touch()
    (tmp_path / "image.png").touch()

    chunks = ing.ingest_dir(tmp_path)

    assert parser.paths == [tmp_path / "b.md", tmp_path / "sub" / "a.md"]
    assert [c.metadata["source"] for c in chunks] == ["b.md", "a.md"]


def test_write_jsonl_round_trip(ingestor_mod, tmp_path):
    chunks = [Chunk("id1", "héllo", {"page": 1}), Chunk("id2", "second\nline", {})]
    out = tmp_path / "nested" / "dir" / "chunks.jsonl"
    ingestor_mod.write_jsonl(chunks, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert [Chunk(**json.loads(ln)) for ln in lines] == chunks
    assert "héllo" in lines[0]  # ensure_ascii=False
