"""Ask the book: retrieved passages + an LLM answer that cites them.

    uv run python -m Interface_03.app --chunks data/book_chunks.jsonl --retriever bge-small \
        --confidence data/eval_results/bge-small.confidence.json

    uv run python -m Interface_03.app --chunks data/book_chunks.jsonl --retriever bm25 --no-llm   # search only

The UI only relies on the `Retriever` interface (search + sources); retrievers are picked from the
same registry the evaluation uses. --confidence shows a calibrated "how likely is a relevant
passage among the results" next to each search; it must come from the same retriever.
"""

from __future__ import annotations

import argparse
import html
import time

from typing import Iterator

import gradio as gr

from Encoder_02.retriever import Hit, Retriever, load_chunks
from Evaluation_04.confidence import Confidence, ConfidenceModel
from Evaluation_04.run import RETRIEVERS
from Generation_05.answerer import Answer, RAGAnswerer
from Generation_05.ask import add_llm_args, make_llm

ALL_SOURCES = "All documents"

CSS = """
.hit { border: 1px solid var(--block-border-color); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
.hit-head { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.hit-rank { font-weight: 700; }
.hit-score { font-family: var(--font-mono); color: var(--color-accent); }
.hit-section { font-weight: 600; }
.hit-meta { color: var(--body-text-color-subdued); font-size: 0.85em; margin: 2px 0 8px; }
.hit-text { white-space: pre-wrap; max-height: 220px; overflow-y: auto; font-size: 0.92em; line-height: 1.45; }
.no-hits { color: var(--body-text-color-subdued); padding: 12px 0; }
.hit-cited { border-color: var(--color-accent); }
.cited-badge { font-size: 0.75em; font-weight: 600; color: var(--color-accent); border: 1px solid var(--color-accent);
               border-radius: 999px; padding: 0 8px; }
"""

LEVEL_ICONS = {"high": "🟢", "medium": "🟡", "low": "🔴"}


def render_confidence(conf: Confidence, k: int) -> str:
    return (f"{LEVEL_ICONS.get(conf.level, '')} **Retrieval confidence: {conf.level}** "
            f"({conf.probability:.0%} chance a relevant passage is in the top {k})")

EXAMPLES = [
    "How does TCP Reno react to a timeout?",
    "What is the difference between link-state and distance-vector routing?",
    "Why do switches forward frames faster than routers?",
]


def render_hits(hits: list[Hit], cited: set[int] = frozenset()) -> str:
    if not hits:
        return '<div class="no-hits">No matching chunks.</div>'
    cards = []
    for rank, h in enumerate(hits, 1):
        m = h.chunk.metadata
        section = html.escape(m.get("section") or "(no heading)")
        meta = " · ".join(html.escape(str(x)) for x in [
            m.get("source", ""), f"p. {m['page']}" if "page" in m else "",
            f"part {m['part']}" if "part" in m else "", f"id {h.chunk.id}",
        ] if x)
        badge = '<span class="cited-badge">cited</span>' if rank in cited else ""
        cards.append(
            f'<div class="hit{" hit-cited" if rank in cited else ""}">'
            f'<div class="hit-head"><span class="hit-rank">[{rank}]</span>'
            f'<span class="hit-score">{h.score:.3f}</span><span class="hit-section">{section}</span>{badge}</div>'
            f'<div class="hit-meta">{meta}</div>'
            f'<div class="hit-text">{html.escape(h.chunk.text)}</div>'
            f'</div>'
        )
    return "\n".join(cards)


def render_answer(ans: Answer) -> str:
    if not ans.done and not ans.text:
        return "_Reading the sources…_"
    text = ans.text + ("" if ans.done else " ▌")
    if ans.done and ans.invalid_citations:
        text += f"\n\n⚠️ _The answer cites sources that don't exist: {ans.invalid_citations}_"
    if ans.done and ans.llm_used and not ans.cited:
        text += "\n\n⚠️ _The answer cites no source; check it against the passages below._"
    return text


def build_app(retriever: Retriever, name: str = "", confidence: ConfidenceModel | None = None,
              answerer: RAGAnswerer | None = None) -> gr.Blocks:
    def ask(query: str, k: int, source: str) -> Iterator[tuple[str, str, str]]:
        """Streams (answer, sources, status)."""
        if not query.strip():
            yield "", "", "Type a question and press Enter."
            return
        t0 = time.perf_counter()
        ans = None
        try:
            for ans in answerer.stream(query, k=int(k), source=None if source == ALL_SOURCES else source):
                sent = f" (first {len(ans.context)} sent to the LLM)" if len(ans.context) < len(ans.hits) else ""
                status = f"{len(ans.hits)} sources{sent} · {time.perf_counter() - t0:.0f} s · {name} + {answerer.llm.name}"
                if ans.confidence:
                    status = render_confidence(ans.confidence, confidence.k) + "  \n" + status
                yield render_answer(ans), render_hits(ans.hits, set(ans.cited)), status
        except Exception as e:  # e.g. Ollama not running; keep the sources that were found
            hits = render_hits(ans.hits) if ans else ""
            yield f"⚠️ Couldn't get an answer from the LLM: `{type(e).__name__}: {e}`  \nIs Ollama running (`ollama serve`)?", hits, ""

    def search(query: str, k: int, source: str) -> tuple[str, str]:
        if not query.strip():
            return "", "Type a question and press Enter."
        k = int(k)
        depth = max(k, confidence.k) if confidence else k  # confidence always looks at its own top k
        t0 = time.perf_counter()
        hits = retriever.search(query, k=depth, source=None if source == ALL_SOURCES else source)
        ms = (time.perf_counter() - t0) * 1000
        status = f"{len(hits[:k])} result{'s' * (len(hits[:k]) != 1)} · {ms:.0f} ms" + (f" · {name}" if name else "")
        if confidence:
            status = render_confidence(confidence.assess(hits), confidence.k) + "  \n" + status
        return render_hits(hits[:k]), status

    title = "NetLLM assistant" if answerer else "NetLLM search"
    with gr.Blocks(title=title) as app:
        gr.Markdown(f"## {title}\n" + ("Answers are generated from the book passages shown below, with citations."
                                       if answerer else "Retrieve book passages for a question."))
        with gr.Row():
            query = gr.Textbox(placeholder="Ask a networking question…", show_label=False, scale=5, autofocus=True)
            button = gr.Button("Ask" if answerer else "Search", variant="primary", scale=1)
        with gr.Row():
            k = gr.Slider(1, 20, value=5, step=1, label="Sources (top-k)")
            source = gr.Dropdown([ALL_SOURCES, *retriever.sources()], value=ALL_SOURCES, label="Document")
        status = gr.Markdown()
        answer = gr.Markdown(visible=answerer is not None)
        results = gr.HTML()
        gr.Examples(EXAMPLES, inputs=query)

        for trigger in (query.submit, button.click):
            if answerer:
                trigger(ask, inputs=[query, k, source], outputs=[answer, results, status])
            else:
                trigger(search, inputs=[query, k, source], outputs=[results, status])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", default="data/book_chunks.jsonl", help="chunks JSONL from the ingestor")
    ap.add_argument("--retriever", choices=sorted(RETRIEVERS), default="bge-small")
    ap.add_argument("--confidence", help="ConfidenceModel JSON from `Evaluation_04.run --calibrate`")
    ap.add_argument("--no-llm", action="store_true", help="search only, no generated answer")
    add_llm_args(ap)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    retriever = RETRIEVERS[args.retriever](chunks)
    print(f"{len(chunks)} chunks loaded from {args.chunks}, retriever: {args.retriever}")

    confidence = ConfidenceModel.load(args.confidence) if args.confidence else None
    if confidence and confidence.retriever != args.retriever:
        raise SystemExit(f"{args.confidence} was calibrated for '{confidence.retriever}', not '{args.retriever}'")
    answerer = None if args.no_llm else RAGAnswerer(retriever, make_llm(args), confidence,
                                                     args.max_context, args.skip_when_low)
    build_app(retriever, args.retriever, confidence, answerer).launch(server_port=args.port, share=args.share, css=CSS)


if __name__ == "__main__":
    main()
