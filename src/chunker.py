from __future__ import annotations

from .models import DocumentChunk, LoadedDocument


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = 1_000,
    chunk_overlap: int = 180,
) -> list[DocumentChunk]:
    """Split loaded documents into overlapping chunks for retrieval."""

    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[DocumentChunk] = []
    for document in documents:
        pieces = _split_text(document.text, chunk_size, chunk_overlap)
        for index, piece in enumerate(pieces, start=1):
            chunk_id = f"{document.source_id}-c{index}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=document.name,
                    text=piece,
                    metadata={**document.metadata, "chunk": index},
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]

        if end < len(text):
            split_at = _best_split_index(window)
            if split_at > chunk_size * 0.55:
                end = start + split_at
                window = text[start:end]

        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return [chunk for chunk in chunks if chunk]


def _best_split_index(window: str) -> int:
    for separator in ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "):
        index = window.rfind(separator)
        if index != -1:
            return index + len(separator)
    return len(window)
