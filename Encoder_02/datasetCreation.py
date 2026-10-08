"""Build a retrieval evaluation set: a local LLM writes questions for sampled book sections.

Pilot (~10 min):  uv run python -m Encoder_02.datasetCreation BOOK.pdf --limit 20
Full  (~1 h):     uv run python -m Encoder_02.datasetCreation BOOK.pdf --limit 150

Sections are visited in a fixed shuffled order, so re-running with a larger --limit (or after a
crash / Ctrl-C) resumes: sections already processed are skipped and the output file is appended to.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import Ingestion_01
from Ingestion_01 import Cleaner, Ingestor, SectionSplitter
from Ingestion_01.doc_dataclass import Section
from Evaluation_04.matching import evidence_ok, norm

QUESTION_TYPES = ["factual", "conceptual", "keyword"]

SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {"query": {"type": "string"},
                       "type": {"type": "string", "enum": QUESTION_TYPES},
                       "evidence": {"type": "string"}},
        "required": ["query", "type", "evidence"]}}},
    "required": ["questions"],
}

PROMPT = """You are building an evaluation set for a retrieval system over a computer networking textbook.
Students will ask questions, and the system must find the passage of the book that answers them.

Below is one section of the book. Write questions that this section answers.

<section_path>{section_path}</section_path>
<section_text>
{section_text}
</section_text>

Rules:
1. Every question must be fully answerable from THIS section alone, and the answer must not
   depend on figures, tables or equations that are missing from the text.
2. Write the way a student studying networking would ask, not like an exam that quotes the book:
   - Paraphrase. Do not copy distinctive phrases or sentence fragments from the text.
   - Never refer to "this section", "the text", "the author", "the figure" or "the example above".
     Each question must make sense on its own, with no context.
3. Mix question types and label each one:
   - "factual": a specific fact, value, definition or behaviour
     (e.g. "What happens to the congestion window after a timeout in TCP Reno?")
   - "conceptual": why or how something works, trade-offs, comparisons
     (e.g. "Why does TCP need its own flow control when the network already has congestion control?")
   - "keyword": built around an exact technical term, acronym, field name or protocol that a
     student would type verbatim (e.g. "What is the purpose of the RcvWindow field?")
4. Write 1 question per ~150 words of section text, at most 5. Prefer fewer good questions to
   many shallow ones.
5. If the section is not real content (exercises/problems list, references, index, table of
   contents, preface, acknowledgements, author bios, or just a heading with a sentence or two),
   return an empty list.
6. For each question, copy the single sentence from the section that best supports the answer,
   VERBATIM (exact characters). It is used to check the retrieved passage automatically.

Return ONLY a JSON object: {"questions": [{"query": "...", "type": "factual|conceptual|keyword", "evidence": "<exact sentence from the section>"}]}
Use {"questions": []} if the section is not real content."""

# Headings of sections that are not explanatory text; skipping them up front saves LLM calls.
# Chapter summaries are skipped too: their questions would also be answered by the original section.
SKIP_HEADINGS = re.compile(
    r"\b(references|bibliography|index|contents|preface|acknowledg\w*|about the authors?|homework|"
    r"problems|exercises|review questions|wireshark lab\w*|programming assignments?|interview|summary)\b",
    re.IGNORECASE,
)

LLM = Callable[[str], list[dict]]


# ------------------------------------------------------------------------- checks

# norm() / evidence_ok() live in Evaluation_04.matching so the metrics use the same rule.

def section_key(source: str, sec: Section) -> str:
    # Page is part of the key because the same heading path can occur more than once.
    return f"{source}|{sec.page}|{' > '.join(sec.path)}"


def select_sections(sections: list[Section], min_words: int, max_words: int, seed: int) -> list[Section]:
    """Content sections in a fixed shuffled order; any --limit takes a prefix, so runs nest."""
    keep = [s for s in sections
            if min_words <= len(s.text.split()) <= max_words
            and not any(SKIP_HEADINGS.search(h) for h in s.path)]
    random.Random(seed).shuffle(keep)
    return keep


# ------------------------------------------------------------------------ loading

def load_sections(book: Path, cache: Path) -> list[Section]:
    """Parse + clean + split once; parsing a whole textbook takes minutes.

    The cache is rebuilt when the book or the ingestion code (parser/cleaner/splitter) changes.
    """
    sources = [book, *Path(Ingestion_01.__file__).parent.glob("*.py")]
    if cache.exists() and cache.stat().st_mtime > max(p.stat().st_mtime for p in sources):
        return [Section(**s) for s in json.loads(cache.read_text(encoding="utf-8"))]
    doc = Cleaner().clean(Ingestor(chunker=object()).load(book))  # chunker unused: skip the tokenizer
    sections = SectionSplitter().split(doc)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(s) for s in sections], ensure_ascii=False), encoding="utf-8")
    return sections


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- llm

def ollama_llm(model: str, think: bool | str, num_ctx: int = 8192, temperature: float = 0.7) -> LLM:
    import ollama

    def call(prompt: str) -> list[dict]:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=SCHEMA,  # constrained decoding: output always parses
            think=think,    # gpt-oss: "low"/"medium"/"high"; Qwen3: True/False
            options={"temperature": temperature, "num_ctx": num_ctx},
        )
        return json.loads(resp.message.content)["questions"]

    return call


# ------------------------------------------------------------------------ generate

def generate(sections: list[Section], llm: LLM, source: str, out: Path, limit: int,
             threshold: float = 90, retries: int = 1, log: Callable[[str], None] = print) -> dict:
    """Ask the LLM about the first `limit` sections not yet processed; append results to `out`.

    Side files next to `out`:  *.progress.jsonl  one row per finished section (resume + stats)
                               *.rejected.jsonl  questions that failed a check, for inspection
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = out.with_suffix(".progress.jsonl")
    rejected = out.with_suffix(".rejected.jsonl")

    done = {r["key"] for r in read_jsonl(progress)}
    seen = {norm(r["query"]) for r in read_jsonl(out)}
    todo = [s for s in sections[:limit] if section_key(source, s) not in done]
    stats = {"sections": 0, "kept": 0, "rejected": 0, "failed": 0, "already_done": min(limit, len(sections)) - len(todo)}
    log(f"{stats['already_done']} sections already done, {len(todo)} to go")

    for i, sec in enumerate(todo, 1):
        path = " > ".join(sec.path)
        prompt = PROMPT.replace("{section_path}", path).replace("{section_text}", sec.text)
        t0 = time.time()
        for attempt in range(retries + 1):
            try:
                questions = llm(prompt)
                break
            except Exception as e:  # bad JSON, Ollama hiccup, missing keys...
                log(f"  [{i}/{len(todo)}] attempt {attempt + 1} failed: {type(e).__name__}: {e}")
        else:
            stats["failed"] += 1  # not marked done, so the next run retries it
            continue

        kept, bad = [], []
        for q in questions:
            query = " ".join(str(q.get("query", "")).split())
            row = {"query": query, "type": q.get("type"), "evidence": q.get("evidence", ""),
                   "relevant": [{"source": source, "section": path, "page": sec.page}]}
            if not query or row["type"] not in QUESTION_TYPES:
                bad.append({**row, "reason": "malformed"})
            elif not evidence_ok(row["evidence"], sec.text, threshold):
                bad.append({**row, "reason": "evidence not in section"})
            elif norm(query) in seen:
                bad.append({**row, "reason": "duplicate"})
            else:
                seen.add(norm(query))
                kept.append(row)

        # Write results before the progress row: a crash in between re-asks the section
        # (its duplicates are then rejected) instead of silently losing its questions.
        append_jsonl(out, kept)
        append_jsonl(rejected, bad)
        secs = round(time.time() - t0, 1)
        append_jsonl(progress, [{"key": section_key(source, sec), "kept": len(kept), "rejected": len(bad), "seconds": secs}])

        stats["sections"] += 1
        stats["kept"] += len(kept)
        stats["rejected"] += len(bad)
        log(f"  [{i}/{len(todo)}] {secs:5.1f}s  +{len(kept)} kept  {len(bad)} rejected  | {path or '(no heading)'}")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book", type=Path, help="PDF / .md / .txt to generate questions for")
    ap.add_argument("--out", type=Path, default=Path("data/eval_questions.jsonl"))
    ap.add_argument("--limit", type=int, default=20, help="total sections to cover (20 = pilot)")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--think", default="low", help='"low"/"medium"/"high" for gpt-oss, "true"/"false" for Qwen3')
    ap.add_argument("--min-words", type=int, default=120)
    ap.add_argument("--max-words", type=int, default=3500, help="longer sections overflow num_ctx")
    ap.add_argument("--threshold", type=float, default=90, help="fuzzy evidence match, 0-100")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    think = {"true": True, "false": False}.get(args.think.lower(), args.think)
    cache = args.out.parent / f"{args.book.stem}.sections.json"

    t0 = time.time()
    all_sections = load_sections(args.book, cache)
    sections = select_sections(all_sections, args.min_words, args.max_words, args.seed)
    print(f"{len(all_sections)} sections parsed, {len(sections)} eligible ({time.time() - t0:.0f}s)")

    stats = generate(sections, ollama_llm(args.model, think), args.book.name, args.out, args.limit, args.threshold)
    total = len(read_jsonl(args.out))
    print(f"\nthis run: {stats}\n{total} questions in {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
