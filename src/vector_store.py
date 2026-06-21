from __future__ import annotations

from pathlib import Path

from .embeddings import cosine_similarity
from .models import DocumentChunk, RetrievedChunk


class VectorStoreError(RuntimeError):
    """Raised when the vector database cannot be initialized or queried."""


class ChromaVectorStore:
    """Disk-persistent ChromaDB vector store for the RAG index."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        reset_collection: bool = False,
        require_existing: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency message for runtime.
            raise VectorStoreError("Install chromadb to use the persistent vector store.") from exc

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection_name = collection_name

        if reset_collection:
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass

        if require_existing:
            try:
                self._collection = self._client.get_collection(collection_name)
            except Exception as exc:
                raise VectorStoreError(
                    f"Chroma collection '{collection_name}' was not found. Run ingest.py first."
                ) from exc
            return

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector.")
        if not chunks:
            return

        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_metadata_for_chroma(chunk) for chunk in chunks],
        )

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            metadata = dict(metadata or {})
            document_name = str(metadata.pop("document_name", "uploaded-document"))
            score = 1.0 - float(distance)
            retrieved.append(
                RetrievedChunk(
                    chunk=DocumentChunk(
                        chunk_id=str(chunk_id),
                        document_name=document_name,
                        text=str(text),
                        metadata=metadata,
                    ),
                    score=score,
                    rank=rank,
                )
            )
        return retrieved

    @property
    def count(self) -> int:
        return self._collection.count()

    @property
    def collection_name(self) -> str:
        return self._collection_name


class InMemoryVectorStore:
    """Small vector store suitable for uploaded documents in a web session."""

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector.")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        scored = [
            (cosine_similarity(query_vector, vector), chunk)
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            RetrievedChunk(chunk=chunk, score=score, rank=index)
            for index, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    @property
    def count(self) -> int:
        return len(self._chunks)


def _metadata_for_chroma(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {"document_name": chunk.document_name}
    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif value is not None:
            metadata[key] = str(value)
    return metadata
