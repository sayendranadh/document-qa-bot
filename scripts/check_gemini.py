from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.embeddings import GeminiEmbeddingProvider
from src.generator import GeminiAnswerGenerator
from src.models import DocumentChunk, RetrievedChunk


def main() -> None:
    load_dotenv()
    config = AppConfig.from_environment()

    if not config.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is missing. Add it to .env or your deployment secrets.")

    print("Checking Gemini embeddings...", flush=True)
    print(f"Embedding model: {config.embedding_model}", flush=True)
    print(f"API key present: yes, length={len(config.gemini_api_key)}", flush=True)

    provider = GeminiEmbeddingProvider(
        api_key=config.gemini_api_key,
        model=config.embedding_model,
    )
    vector = provider.embed_query("health check for document question answering")

    print("Gemini embedding request succeeded.", flush=True)
    print(f"Embedding dimensions: {len(vector)}", flush=True)

    print("Checking Gemini answer generation...", flush=True)
    generator = GeminiAnswerGenerator(
        api_key=config.gemini_api_key,
        model=config.chat_model,
    )
    answer = generator.generate(
        "What should this diagnostic answer say?",
        [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="diagnostic-1",
                    document_name="diagnostic.txt",
                    text="The diagnostic answer should say that Gemini generation is working.",
                    metadata={"chunk": 1},
                ),
                score=1.0,
                rank=1,
            )
        ],
    )
    print("Gemini generation request succeeded.", flush=True)
    print(answer, flush=True)


if __name__ == "__main__":
    main()
