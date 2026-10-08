"""Retrieval-augmented answering: retrieve -> (confidence) -> prompt -> stream the LLM answer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator

from Encoder_02.retriever import Hit, Retriever
from Evaluation_04.confidence import Confidence, ConfidenceModel

from .llm import ChatLLM
from .prompt import build_messages, extract_citations

NO_HITS = "I couldn't find anything about this in the book."
LOW_CONFIDENCE = ("The book doesn't seem to cover this question (retrieval confidence is low), "
                  "so I won't guess. Have a look at the closest passages below or rephrase the question.")


@dataclass
class Answer:
    question: str
    text: str = ""
    hits: list[Hit] = field(default_factory=list)      # shown to the user (top k)
    context: list[Hit] = field(default_factory=list)   # sent to the LLM, numbered [1]..[n]
    confidence: Confidence | None = None
    cited: list[int] = field(default_factory=list)     # valid source numbers the answer cites
    invalid_citations: list[int] = field(default_factory=list)
    llm_used: bool = False
    done: bool = False


class RAGAnswerer:
    """`max_context` caps how many chunks go into the prompt (each is up to ~500 tokens; keep the
    prompt well inside the LLM's num_ctx). With `skip_when_low`, a low retrieval confidence
    answers "not covered" without calling the LLM."""

    def __init__(self, retriever: Retriever, llm: ChatLLM, confidence: ConfidenceModel | None = None,
                 max_context: int = 6, skip_when_low: bool = False):
        self.retriever = retriever
        self.llm = llm
        self.confidence = confidence
        self.max_context = max_context
        self.skip_when_low = skip_when_low

    def retrieve(self, question: str, k: int = 5, source: str | None = None) -> tuple[list[Hit], Confidence | None]:
        depth = max(k, self.confidence.k) if self.confidence else k  # confidence judges its own top k
        hits = self.retriever.search(question, k=depth, source=source)
        conf = self.confidence.assess(hits) if self.confidence else None
        return hits[:k], conf

    def stream(self, question: str, k: int = 5, source: str | None = None) -> Iterator[Answer]:
        """Yields the growing answer; the last item has done=True and the citation check."""
        hits, conf = self.retrieve(question, k, source)
        ans = Answer(question, hits=hits, context=hits[: self.max_context], confidence=conf)
        if not hits:
            yield replace(ans, text=NO_HITS, done=True)
            return
        if self.skip_when_low and conf is not None and conf.level == "low":
            yield replace(ans, text=LOW_CONFIDENCE, done=True)
            return

        yield ans  # retrieval done: the UI can show the sources while the LLM starts
        text = ""
        for piece in self.llm.stream(build_messages(question, ans.context)):
            text += piece
            yield replace(ans, text=text, llm_used=True)
        cited, invalid = extract_citations(text, len(ans.context))
        yield replace(ans, text=text.strip(), llm_used=True, cited=cited, invalid_citations=invalid, done=True)

    def answer(self, question: str, k: int = 5, source: str | None = None) -> Answer:
        *_, last = self.stream(question, k, source)
        return last
