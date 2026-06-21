from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency is loaded in app runtime.
    load_dotenv = None


if load_dotenv:
    load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Runtime knobs for the RAG pipeline."""

    gemini_api_key: str | None = None
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    chunk_size: int = 1_000
    chunk_overlap: int = 180
    top_k: int = 5
    min_score: float = 0.10
    persist_directory: str = "db"
    show_progress: bool = True

    @classmethod
    def from_environment(cls) -> "AppConfig":
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "db"),
        )
