from __future__ import annotations

from .doc_dataclass import Document, Section, Chunk
import hashlib
import re


# ------------------------------------------------------------------------ chunker

class Chunker:
    """Size split: packs paragraphs into chunks that fit the embedding model.

    Token counts use the embedding model's own tokenizer. bge models truncate at 512
    tokens, so max_tokens + overlap stays below that.
    """

    def __init__(self, tokenizer_name: str = "BAAI/bge-small-en-v1.5",
                 max_tokens: int = 450, overlap_tokens: int = 50, min_tokens: int = 30):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def n_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def chunk(self, doc: Document, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for sec in sections:
            header = " > ".join(sec.path)
            # Prefixing the heading path gives each chunk context ("which section am I?").
            prefix = f"{header}\n\n" if header else ""
            budget = self.max_tokens - self.n_tokens(prefix)
            for part, body in enumerate(self._pack(sec.text, budget)):
                if self.n_tokens(body) < self.min_tokens:
                    continue
                text = prefix + body
                chunks.append(Chunk(
                    # Deterministic id: re-ingesting the same file upserts instead of duplicating.
                    id=hashlib.sha1(f"{doc.source}|{header}|{part}|{text}".encode()).hexdigest()[:16],
                    text=text,
                    metadata={"source": doc.source, "section": header, "page": sec.page, "part": part},
                ))
        return chunks

    def _pack(self, text: str, budget: int) -> list[str]:
        units: list[str] = []
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            units.extend([para] if self.n_tokens(para) <= budget else self._split_long(para, budget))

        out: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for u in units:
            n = self.n_tokens(u)
            if cur and cur_len + n > budget:
                out.append("\n\n".join(cur))
                # Overlap: carry the last paragraph forward if it is short.
                tail = cur[-1]
                cur, cur_len = ([tail], self.n_tokens(tail)) if self.n_tokens(tail) <= self.overlap_tokens else ([], 0)
            cur.append(u)
            cur_len += n
        if cur:
            out.append("\n\n".join(cur))
        return out

    def _split_long(self, para: str, budget: int) -> list[str]:
        """Oversized paragraph: split on sentences; hard-cut only as a last resort (tables)."""
        pieces: list[str] = []
        cur = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            cand = f"{cur} {sent}".strip()
            if cur and self.n_tokens(cand) > budget:
                pieces.append(cur)
                cur = sent
            else:
                cur = cand
        if cur:
            pieces.append(cur)

        final: list[str] = []
        for piece in pieces:
            while self.n_tokens(piece) > budget:
                cut = max(1, len(piece) * budget // self.n_tokens(piece))
                # The estimate assumes uniform chars/token; shrink until the head really fits.
                while cut > 1 and (n := self.n_tokens(piece[:cut])) > budget:
                    cut = max(1, min(cut - 1, cut * budget // n))
                final.append(piece[:cut])
                piece = piece[cut:]
            if piece.strip():
                final.append(piece)
        return final

