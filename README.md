# NetLLM-small-proto

Small prototype of NetLLM: a retrieval-augmented (RAG) assistant for computer networking that
answers questions from a textbook (*Computer Networking: A Top-Down Approach*, Kurose & Ross),
citing the passages it used.

```
PDF ─► parse ─► clean ─► split ─► chunk ─► chunks.jsonl ─► embed (bge) ─► Chroma ─► retrieve ─► LLM answer
       └──────────── Ingestion_01 ────────────┘              └──────── Encoder_02 ────────┘   Generation_05
                                     Evaluation_04: question set, metrics, baselines, confidence
                                     Interface_03:  Gradio app
```

Everything runs locally: embeddings with `BAAI/bge-small-en-v1.5`, vectors in Chroma, answers and
question generation with `gpt-oss:20b` through Ollama.

---

## 1. Prerequisites

| Tool | Why | Install |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python 3.14 + dependencies | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Ollama](https://ollama.com/download) | Local LLM (answers + dataset generation) | see the website |
| NVIDIA GPU (optional) | Faster embeddings / LLM | Everything also runs on CPU, more slowly |

Hardware used during development: 6 GB VRAM GPU, 32 GB RAM. `gpt-oss:20b` needs ~14 GB of RAM + VRAM
combined (it is split between CPU and GPU automatically).

## 2. Setup

```bash
git clone <repo-url> NetLLM-small-proto
cd NetLLM-small-proto

uv sync                      # creates .venv with Python 3.14, installs dependencies and this project (editable)
ollama pull gpt-oss:20b      # ~13 GB, once
```

- `uv sync` installs the project packages (`Ingestion_01`, `Encoder_02`, …), so they can be imported from
  anywhere, including the notebooks (`from Ingestion_01 import Ingestor`).
- The bge model (~130 MB) is downloaded from Hugging Face automatically on first use.
- The textbook PDF is **not** in the repository. All commands below use `$BOOK` for its path:

```bash
export BOOK=/path/to/ComputerNetworking.pdf
```

> All commands are run **from the project root** with `uv run python -m <package>.<module>`
> (module form, not file paths).

## 3. Build the knowledge base

### 3.1 Ingest the book → chunks (~3 min)

```bash
uv run python -m Ingestion_01.ingestor "$BOOK" --out data/book_chunks.jsonl
```

Parses the PDF to Markdown, removes headers/footers/page numbers, splits it into sections (one per
heading, keeping the heading path), and packs the text into ≤ 450-token chunks. Output: one JSON chunk
per line, e.g. ~1,700 chunks for the book. `input` can also be a folder (all `.pdf`, `.md`, `.txt` inside).

### 3.2 Embed and index the chunks in Chroma (~30 s on GPU)

```bash
uv run python -m Encoder_02.indexer data/book_chunks.jsonl
```

| Option | Default | |
|---|---|---|
| `--db` | `data/chroma` | Chroma folder |
| `--model` | `BAAI/bge-small-en-v1.5` | embedding model (one collection per model) |
| `--device` | `auto` | `cpu` / `cuda` (use `cpu` if the GPU is busy or out of memory) |
| `--remove SOURCE` | | remove a document, e.g. `--remove NetLLM_paper.pdf` |

Re-running is cheap: only new or changed chunks are embedded, stale ones are deleted, and other
documents in the index are left alone. Several files can be indexed into the same database.

## 4. Generate the evaluation dataset (DatasetGen)

An LLM writes questions about book sections; each question keeps a verbatim **evidence** sentence and
its **section label**, so retrieval can be scored automatically. Ollama must be running.

```bash
# Pilot: 20 sections (~10 min). Read data/eval_questions.jsonl before going further.
uv run python -m Encoder_02.datasetCreation "$BOOK" --limit 20

# Full run: all eligible sections (443 for the book, ~4 h). Continues where the pilot stopped.
uv run python -m Encoder_02.datasetCreation "$BOOK" --limit 443
```

- **Resumable:** sections are visited in a fixed shuffled order; Ctrl-C, crashes or a larger `--limit`
  simply continue (finished sections are skipped).
- **Which sections:** every section with 120–3,500 words whose heading path does not look like
  references, index, homework/problems, labs, interviews or summaries (`SKIP_HEADINGS`).
- **Checks:** evidence must be found in the section (fuzzy match), valid type, no duplicates.

| Option | Default | |
|---|---|---|
| `--out` | `data/eval_questions.jsonl` | dataset |
| `--limit` | `20` | total number of sections to cover |
| `--model` / `--think` | `gpt-oss:20b` / `low` | Ollama model; `--think true/false` for Qwen3 |
| `--min-words` / `--max-words` | `120` / `3500` | section length filter |
| `--threshold` | `90` | fuzzy evidence match (0–100) |
| `--seed` | `0` | section order |

Files written next to `--out`:

| File | Content |
|---|---|
| `eval_questions.jsonl` | `{"query", "type": factual\|conceptual\|keyword, "evidence", "relevant": [{"source", "section", "page"}]}` |
| `eval_questions.progress.jsonl` | one line per finished section (used for resuming, has timings) |
| `eval_questions.rejected.jsonl` | rejected questions and the reason |
| `<book>.sections.json` | parsed-book cache (rebuilt automatically if the PDF or `Ingestion_01` changes) |

## 5. Evaluate the retrievers

```bash
uv run python -m Evaluation_04.run --chunks data/book_chunks.jsonl --retrievers bm25 bge-small --calibrate
```

Prints, for each retriever: Recall@1/5/10/20 and MRR@10 with 95 % bootstrap CIs (resampled by section), under two relevance
definitions (**evidence**: the chunk contains the evidence sentence; **section**: the chunk comes from the
labelled section), a breakdown per question type, and a paired significance test against the first
retriever listed.

| Option | Default | |
|---|---|---|
| `--eval` | `data/eval_questions.jsonl` | question set |
| `--chunks` | *(required)* | chunks of the **same** book the questions come from |
| `--retrievers` | `bm25` | any of `bm25`, `bge-small`, `bm25+rerank`, `bge-small+rerank`, `stub` (first = reference) |
| `--mode` | `both` | `evidence`, `section` or `both` |
| `--ks` / `--mrr-k` / `--primary-k` | `1 5 10 20` / `10` / `5` | cut-offs |
| `--out` | `data/eval_results` | per-query results (`<retriever>.jsonl`) for error analysis |
| `--calibrate` | off | also fit a **runtime confidence model** per retriever → `<retriever>.confidence.json` |
| `--ci` | `section` | confidence intervals by resampling whole sections, or single questions |

**Re-report without re-running** (seconds): every run saves per-question results; the tables and the
confidence calibration can be recomputed from them, e.g. with other cut-offs or CI method:

```bash
uv run python -m Evaluation_04.report data/eval_results/bge-small.jsonl data/eval_results/bge-small+rerank.jsonl
uv run python -m Evaluation_04.report data/eval_results --calibrate     # all results files in the folder
```

The first file is the reference for comparisons. Options: `--mode`, `--ks`, `--mrr-k`, `--primary-k`,
`--ci section|question` (default `section`: questions about the same section are resampled together,
since they succeed or fail together), `--calibrate`, `--out`.

**Reranking:** the `+rerank` retrievers take the base retriever's top 50 candidates and re-order them
with a cross-encoder (`BAAI/bge-reranker-base`, ~1.1 GB, downloaded on first use), which reads question
and passage together. ~1 s per question on a 6 GB GPU (~30 min for 1,620 questions); settings in
`RERANKER` / `RERANK_CANDIDATES` in `Evaluation_04/run.py`.

The confidence model maps label-free signals of a search (how much the best result stands out) to the
probability that a relevant passage is among the results; the app shows it as a 🟢/🟡/🔴 badge.
Refit it (`--calibrate`) whenever the retriever, the model or the chunking changes.

## 6. Run the app

```bash
uv run python -m Interface_03.app --confidence data/eval_results/bge-small.confidence.json
```

Open <http://127.0.0.1:7860>. Ask a question: the answer streams in, citing sources as `[1]`, `[2]`, …;
the retrieved passages are shown below it (cited ones highlighted), with the retrieval confidence on top.

| Option | Default | |
|---|---|---|
| `--chunks` | `data/book_chunks.jsonl` | chunks to search |
| `--retriever` | `bge-small` | `bge-small`, `bm25`, `bge-small+rerank`, `bm25+rerank` or `stub` |
| `--confidence` | | confidence JSON of the **same** retriever (from step 5) |
| `--no-llm` | off | search only, no generated answer |
| `--llm` / `--think` | `gpt-oss:20b` / `low` | Ollama model |
| `--max-context` | `6` | chunks sent to the LLM |
| `--skip-when-low` | off | don't call the LLM when retrieval confidence is low |
| `--port` / `--share` | `7860` / off | port; `--share` creates a temporary public link |

The first answer takes ~1–2 min (the LLM is loaded); afterwards ~20–40 s. The model stays loaded for
30 min after the last question.

**Terminal version** (same options):

```bash
uv run python -m Generation_05.ask "How does TCP Reno react to a timeout?" \
    --confidence data/eval_results/bge-small.confidence.json
```

## 7. Unit tests

```bash
uv run pytest unit_tests -q                        # all (~10 s)
uv run pytest unit_tests/test_splitter.py -q       # one file
uv run pytest unit_tests -q -k confidence          # tests matching a name
```

The tests are fully offline: no Ollama, no model download (fake tokenizer/LLM/embedder, a tiny random
BERT, and a real Chroma database in a temporary folder).

| File | Covers |
|---|---|
| `test_cleaner.py`, `test_splitter.py`, `test_chunker.py`, `test_parsers.py`, `test_ingestor.py` | ingestion pipeline (incl. the paper's and the textbook's heading layouts) |
| `test_dataset_creation.py` | DatasetGen: evidence check, section selection, resume, dedup, retries, cache |
| `test_encoder.py` | embedder (pooling, normalisation, query instruction), Chroma sync, retrievers |
| `test_reranker.py` | cross-encoder scoring (1/2 labels, truncation, batching), reranking wrapper |
| `test_evaluation.py` | relevance, metrics, bootstrap (incl. by section), BM25, confidence calibration, run/report CLIs |
| `test_generation.py` | prompt, citations, streaming answers, LLM errors |
| `test_interface.py` | Gradio app wiring and rendering |

## Project structure

| Package | Content |
|---|---|
| `Ingestion_01/` | `parsers` (PDF/MD/TXT) · `cleaner` · `splitter` (heading paths) · `chunker` (token-based) · `ingestor` (CLI) |
| `Encoder_02/` | `embedder` (HF bi-encoder) · `indexer` (Chroma sync, CLI) · `retriever` (`Retriever` contract, stub, bi-encoder) · `reranker` (cross-encoder + `RerankingRetriever`) · `datasetCreation` (DatasetGen CLI) |
| `Interface_03/` | `app` (Gradio) |
| `Evaluation_04/` | `matching` · `dataset` · `metrics` · `baselines` (BM25) · `evaluate` · `confidence` · `run` (CLI + retriever registry) |
| `Generation_05/` | `prompt` · `llm` (Ollama) · `answerer` (RAG) · `ask` (CLI) |
| `unit_tests/` | pytest suite |
| `notebooks/` | exploration notebooks |
| `data/` | generated files: chunks, Chroma, questions, results (**git-ignored**) |

**Adding a retriever** (e.g. `bge-base`, hybrid): implement `search(query, k, source)` and `sources()`
(see `Encoder_02/retriever.py`) and register it in `RETRIEVERS` in `Evaluation_04/run.py`; the evaluation,
the calibration and the app pick it up automatically.

## Troubleshooting

| Message | Meaning |
|---|---|
| `Token indices sequence length is longer than … (559 > 512)` | Harmless: the chunker *measures* long paragraphs before splitting them; all final chunks are < 512 tokens. |
| `RuntimeWarning: 'Ingestion_01.ingestor' found in sys.modules …` | Harmless: the module is imported by the package before being run as a script. |
| `Couldn't get an answer from the LLM … Is Ollama running?` | Start Ollama (`ollama serve`) and check `ollama list` shows the model. |
| CUDA out of memory | Another model (e.g. gpt-oss) is using the GPU: add `--device cpu` to the indexer. |
| `The questions are about […] but the chunks come from […]` | Evaluation needs the chunks of the same book the questions were generated from. |
| Jupyter kernel disappeared after `uv sync` | `uv sync` removes packages not declared in `pyproject.toml`; add them with `uv add` / `uv add --dev`. |

> **Copyright:** `data/` contains text from the textbook (chunks, evidence sentences). It is git-ignored;
> don't publish it.
