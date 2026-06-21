from __future__ import annotations

from uuid import uuid4

from .chunker import chunk_documents
from .config import AppConfig
from .embeddings import EmbeddingProvider, GeminiEmbeddingProvider, HashEmbeddingProvider
from .generator import AnswerGenerator, ExtractiveAnswerGenerator, GeminiAnswerGenerator
from .models import DocumentChunk, LoadedDocument, RAGAnswer
from .vector_store import ChromaVectorStore


class RAGPipeline:
    """Coordinates chunking, embedding, retrieval, and grounded generation."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        answer_generator: AnswerGenerator,
        mode: str,
        indexed_chunks: list[DocumentChunk],
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.answer_generator = answer_generator
        self.mode = mode
        self.indexed_chunks = indexed_chunks

    @classmethod
    def build(cls, documents: list[LoadedDocument], config: AppConfig) -> "RAGPipeline":
        chunks = chunk_documents(
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks were created from the uploaded documents.")

        if config.gemini_api_key:
            embedding_provider: EmbeddingProvider = GeminiEmbeddingProvider(
                api_key=config.gemini_api_key,
                model=config.embedding_model,
            )
            answer_generator: AnswerGenerator = GeminiAnswerGenerator(
                api_key=config.gemini_api_key,
                model=config.chat_model,
            )
            mode = "Gemini RAG"
        else:
            embedding_provider = HashEmbeddingProvider()
            answer_generator = ExtractiveAnswerGenerator()
            mode = "Offline demo"

        vectors = _embed_with_progress(
            embedding_provider=embedding_provider,
            texts=[chunk.text for chunk in chunks],
            enabled=config.show_progress,
        )
        vector_store = ChromaVectorStore(
            persist_directory=config.persist_directory,
            collection_name=f"rag_{uuid4().hex[:12]}",
            reset_collection=True,
        )
        vector_store.add(chunks, vectors)

        return cls(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            answer_generator=answer_generator,
            mode=mode,
            indexed_chunks=chunks,
        )

    def ask(self, question: str, top_k: int, min_score: float = 0.0) -> RAGAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        query_vector = self.embedding_provider.embed_query(question)
        retrieved_chunks = self.vector_store.search(query_vector, top_k=top_k)
        filtered_chunks = [item for item in retrieved_chunks if item.score >= min_score]
        if not filtered_chunks:
            best_score = retrieved_chunks[0].score if retrieved_chunks else None
            best_score_text = f"{best_score:.3f}" if best_score is not None else "none"
            answer = (
                "No retrieved chunks passed the current similarity threshold. "
                f"The best retrieved score was {best_score_text}, while Minimum similarity is set to {min_score:.2f}. "
                "Lower Minimum similarity to around 0.00-0.20 and ask again."
            )
            return RAGAnswer(
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                mode=self.mode,
                threshold_blocked=True,
            )

        answer = self.answer_generator.generate(question, filtered_chunks)
        return RAGAnswer(answer=answer, retrieved_chunks=filtered_chunks, mode=self.mode)


def _embed_with_progress(
    embedding_provider: EmbeddingProvider,
    texts: list[str],
    enabled: bool,
) -> list[list[float]]:
    if not enabled:
        return embedding_provider.embed_documents(texts)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return embedding_provider.embed_documents(texts)

    vectors: list[list[float]] = []
    for text in tqdm(texts, desc="Embedding chunks", unit="chunk"):
        vectors.extend(embedding_provider.embed_documents([text]))
    return vectors
