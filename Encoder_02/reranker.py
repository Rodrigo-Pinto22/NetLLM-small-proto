"""Second-stage reranking: a cross-encoder reads (query, chunk) together and re-orders candidates.

A bi-encoder or BM25 retrieves many candidates fast (high recall); the cross-encoder scores each
pair jointly (high precision) and only the best k are kept. `RerankingRetriever` wraps any
`Retriever`, so the app, the evaluation and the answer generation use it unchanged.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .retriever import Hit, Retriever


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to the query; higher = more relevant."""
        ...


class CrossEncoderReranker:
    """Hugging Face sequence-classification cross-encoder, e.g. BAAI/bge-reranker-base.

    Scores are raw logits (unbounded): fine for ordering and for the confidence calibrator.
    Models with two labels (not relevant / relevant) use the logit difference.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "auto",
                 batch_size: int = 16, max_length: int = 512, fp16: bool | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if fp16 is None:
            fp16 = device.startswith("cuda")  # halves memory and roughly doubles speed on GPU
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        if fp16:
            self.model.half()

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            # Truncate the chunk, never the question.
            enc = self.tokenizer([query] * len(batch), batch, padding=True, truncation="only_second",
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.float()
            col = logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]
            scores.extend(col.cpu().tolist())
        return scores


class RerankingRetriever:
    """Retrieve `candidates` chunks with `base`, rerank them, return the best k.

    Recall@k can never exceed the base retriever's recall at `candidates`: the reranker only
    re-orders what stage 1 found. Hit scores are the reranker's, not the base retriever's.
    """

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 50):
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self.base = base
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[Hit]:
        cands = self.base.search(query, k=max(k, self.candidates), source=source)
        if not cands:
            return []
        scores = self.reranker.score(query, [h.chunk.text for h in cands])
        # sorted() is stable: equal scores keep the base retriever's order.
        ranked = sorted(zip(cands, scores), key=lambda hs: hs[1], reverse=True)
        return [Hit(h.chunk, s) for h, s in ranked[:k]]

    def sources(self) -> list[str]:
        return self.base.sources()
