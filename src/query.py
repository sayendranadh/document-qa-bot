from __future__ import annotations

import argparse

from src.config import AppConfig
from src.embeddings import EmbeddingProvider, GeminiEmbeddingProvider, HashEmbeddingProvider
from src.generator import AnswerGenerator, ExtractiveAnswerGenerator, GeminiAnswerGenerator
from src.vector_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query an existing ChromaDB RAG index.")
    parser.add_argument("--db-path", default="db", help="Folder containing the persisted ChromaDB index.")
    parser.add_argument("--collection", default="document_qa", help="ChromaDB collection name.")
    parser.add_argument("--question", help="Question to ask. If omitted, interactive mode starts.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--min-score", type=float, help="Minimum similarity score to keep.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline hash embeddings. Use only with a DB created using ingest.py --offline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_environment()

    embedding_provider, answer_generator, mode = _build_providers(config, use_offline=args.offline)
    vector_store = ChromaVectorStore(
        persist_directory=args.db_path,
        collection_name=args.collection,
        require_existing=True,
    )

    print(f"Loaded collection '{vector_store.collection_name}' with {vector_store.count} chunks.")
    print(f"Mode: {mode}")
    min_score = args.min_score
    if min_score is None:
        min_score = -1.0 if args.offline else 0.10

    if args.question:
        _answer_question(
            args.question,
            args.top_k,
            min_score,
            embedding_provider,
            answer_generator,
            vector_store,
        )
        return

    while True:
        question = input("\nAsk a question, or type 'exit': ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            _answer_question(
                question,
                args.top_k,
                min_score,
                embedding_provider,
                answer_generator,
                vector_store,
            )


def _build_providers(
    config: AppConfig,
    use_offline: bool,
) -> tuple[EmbeddingProvider, AnswerGenerator, str]:
    if config.gemini_api_key and not use_offline:
        return (
            GeminiEmbeddingProvider(config.gemini_api_key, config.embedding_model),
            GeminiAnswerGenerator(config.gemini_api_key, config.chat_model),
            "Gemini RAG",
        )

    return HashEmbeddingProvider(), ExtractiveAnswerGenerator(), "offline retrieval demo"


def _answer_question(
    question: str,
    top_k: int,
    min_score: float,
    embedding_provider: EmbeddingProvider,
    answer_generator: AnswerGenerator,
    vector_store: ChromaVectorStore,
) -> None:
    query_vector = embedding_provider.embed_query(question)
    retrieved = vector_store.search(query_vector, top_k=top_k)
    retrieved = [item for item in retrieved if item.score >= min_score]
    answer = answer_generator.generate(question, retrieved)

    print("\nAnswer")
    print(answer)
    print("\nRetrieved Sources")
    if not retrieved:
        print("No chunks passed the similarity threshold.")
        return
    for item in retrieved:
        print(f"[{item.rank}] {item.chunk.citation_label} | score={item.score:.3f}")


if __name__ == "__main__":
    main()
