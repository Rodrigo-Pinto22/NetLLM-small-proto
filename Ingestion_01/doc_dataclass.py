from dataclasses import asdict, dataclass, field


@dataclass
class Page:
    text: str
    number: int  # 1-based; always 1 for formats without pages

@dataclass
class Document:
    source: str
    pages: list[Page]
    metadata: dict = field(default_factory=dict)

@dataclass
class Section:
    path: list[str]  # heading hierarchy, e.g. ["Transport Layer", "3.5 TCP"]
    text: str
    page: int


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict