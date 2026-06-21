from __future__ import annotations

import argparse
from pathlib import Path

from src.chunker import chunk_documents
from src.config import AppConfig
from src.document_loader import scan_document_folder
from src.embeddings import EmbeddingProvider, GeminiEmbeddingProvider, HashEmbeddingProvider
from src.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into a persistent ChromaDB index.")
    parser.add_argument("--documents", default="data", help="Folder containing source documents.")
    parser.add_argument("--db-path", default="db", help="Folder where ChromaDB will persist vectors.")
    parser.add_argument("--collection", default="document_qa", help="ChromaDB collection name.")
    parser.add_argument("--chunk-size", type=int, default=1_000, help="Maximum characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=180, help="Character overlap between chunks.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic local hash embeddings instead of Gemini embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_environment()

    documents_path = Path(args.documents)
    documents = scan_document_folder(documents_path)
    if not documents:
        raise SystemExit(f"No supported documents found in {documents_path}.")

    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    embedding_provider, mode = _build_embedding_provider(config, use_offline=args.offline)
    vectors = _embed_chunks(embedding_provider, [chunk.text for chunk in chunks])

    vector_store = ChromaVectorStore(
        persist_directory=args.db_path,
        collection_name=args.collection,
        reset_collection=True,
    )
    vector_store.add(chunks, vectors)

    print(f"Ingestion complete using {mode}.")
    print(f"Documents loaded: {len({document.name for document in documents})}")
    print(f"Chunks indexed: {vector_store.count}")
    print(f"Chroma path: {Path(args.db_path).resolve()}")
    print(f"Collection: {args.collection}")


def _build_embedding_provider(config: AppConfig, use_offline: bool) -> tuple[EmbeddingProvider, str]:
    if config.gemini_api_key and not use_offline:
        return (
            GeminiEmbeddingProvider(
                api_key=config.gemini_api_key,
                model=config.embedding_model,
            ),
            "Gemini embeddings",
        )

    return HashEmbeddingProvider(), "offline hash embeddings"


def _embed_chunks(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return provider.embed_documents(texts)

    vectors: list[list[float]] = []
    for text in tqdm(texts, desc="Embedding chunks", unit="chunk"):
        vectors.extend(provider.embed_documents([text]))
    return vectors


if __name__ == "__main__":
    main()
