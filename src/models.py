from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoadedDocument:
    """Text extracted from one logical source, such as a page or a text file."""

    source_id: str
    name: str
    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable piece of a source document."""

    chunk_id: str
    document_name: str
    text: str
    metadata: dict[str, str | int]

    @property
    def citation_label(self) -> str:
        page = self.metadata.get("page")
        if page:
            return f"{self.document_name}, Page {page}"
        return self.document_name


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by semantic search, including its relevance score."""

    chunk: DocumentChunk
    score: float
    rank: int


@dataclass(frozen=True)
class RAGAnswer:
    """Final answer returned to the UI."""

    answer: str
    retrieved_chunks: list[RetrievedChunk]
    mode: str
    threshold_blocked: bool = False
