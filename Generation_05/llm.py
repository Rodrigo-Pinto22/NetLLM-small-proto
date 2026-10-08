"""Chat LLM contract + Ollama implementation. Swap in another backend by implementing `stream`."""

from __future__ import annotations

from typing import Iterator, Protocol, Sequence


class ChatLLM(Protocol):
    name: str

    def stream(self, messages: Sequence[dict]) -> Iterator[str]:
        """Yield the answer text piece by piece (no reasoning traces)."""
        ...


class OllamaChat:
    def __init__(self, model: str = "gpt-oss:20b", think: bool | str = "low", temperature: float = 0.2,
                 num_ctx: int = 8192, keep_alive: str = "30m", host: str | None = None):
        import ollama

        self.name = model
        self.think = think            # gpt-oss: "low"/"medium"/"high"; Qwen3: True/False
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self.keep_alive = keep_alive  # keep the model loaded between questions
        self.client = ollama.Client(host=host)

    def stream(self, messages: Sequence[dict]) -> Iterator[str]:
        for part in self.client.chat(model=self.name, messages=list(messages), stream=True, think=self.think,
                                     options=self.options, keep_alive=self.keep_alive):
            if part.message.content:  # reasoning arrives in part.message.thinking and is not shown
                yield part.message.content
