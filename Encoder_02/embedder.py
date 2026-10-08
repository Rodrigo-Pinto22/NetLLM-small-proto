from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

# bge v1.5 expects this prefix on queries only; passages are embedded as they are.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HFEmbedder:
    """Hugging Face encoder with CLS or mean pooling and L2 normalisation.

    Defaults match BAAI/bge-*-en-v1.5 (CLS pooling + query instruction). For e5-style models use
    pooling="mean" and their "query: " / "passage: " prefixes.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "auto",
                 batch_size: int = 32, max_length: int = 512, pooling: str = "cls",
                 query_instruction: str = BGE_QUERY_INSTRUCTION, passage_prefix: str = ""):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', not {pooling!r}")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.query_instruction = query_instruction
        self.passage_prefix = passage_prefix
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(list(texts[i:i + self.batch_size]), padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
            if self.pooling == "cls":
                vec = hidden[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(vec, dim=-1).float().cpu().numpy())
        dim = self.model.config.hidden_size
        return np.concatenate(out) if out else np.empty((0, dim), dtype=np.float32)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_instruction + text])[0]
