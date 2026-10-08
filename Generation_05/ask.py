"""Ask the book a question from the terminal (streams the answer, then lists the sources).

    uv run python -m Generation_05.ask "How does TCP Reno react to a timeout?" \
        --chunks data/book_chunks.jsonl --retriever bge-small \
        --confidence data/eval_results/bge-small.confidence.json
"""

from __future__ import annotations

import argparse
import time

from Encoder_02.retriever import load_chunks
from Evaluation_04.confidence import ConfidenceModel
from Evaluation_04.run import RETRIEVERS

from .answerer import RAGAnswerer
from .llm import OllamaChat


def add_llm_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--llm", default="gpt-oss:20b", help="Ollama model")
    ap.add_argument("--think", default="low", help='"low"/"medium"/"high" for gpt-oss, "true"/"false" for Qwen3')
    ap.add_argument("--max-context", type=int, default=6, help="chunks sent to the LLM")
    ap.add_argument("--skip-when-low", action="store_true", help="don't call the LLM when retrieval confidence is low")


def make_llm(args: argparse.Namespace) -> OllamaChat:
    return OllamaChat(args.llm, think={"true": True, "false": False}.get(args.think.lower(), args.think))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--chunks", default="data/book_chunks.jsonl")
    ap.add_argument("--retriever", choices=sorted(RETRIEVERS), default="bge-small")
    ap.add_argument("--confidence", help="ConfidenceModel JSON for the same retriever")
    ap.add_argument("-k", type=int, default=5)
    add_llm_args(ap)
    args = ap.parse_args()

    retriever = RETRIEVERS[args.retriever](load_chunks(args.chunks))
    confidence = ConfidenceModel.load(args.confidence) if args.confidence else None
    answerer = RAGAnswerer(retriever, make_llm(args), confidence, args.max_context, args.skip_when_low)

    t0, printed, header = time.time(), 0, False
    for ans in answerer.stream(args.question, k=args.k):
        if not header and ans.confidence:
            print(f"[confidence: {ans.confidence.level} {ans.confidence.probability:.0%}]\n")
        header = True
        print(ans.text[printed:], end="", flush=True)
        printed = len(ans.text)
    print(f"\n\n({time.time() - t0:.0f}s)  sources:")
    for i, h in enumerate(ans.context, 1):
        mark = "*" if i in ans.cited else " "
        print(f" {mark}[{i}] {h.score:.3f}  {h.chunk.metadata.get('section', '')}  p.{h.chunk.metadata.get('page', '?')}")
    if ans.invalid_citations:
        print(f" ! cites non-existent sources: {ans.invalid_citations}")


if __name__ == "__main__":
    main()
