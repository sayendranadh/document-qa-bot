from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot embed text."""


class EmbeddingProvider(ABC):
    """Interface for vectorizing document chunks and user queries."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding provider used in the production RAG path."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - dependency message for runtime.
            raise EmbeddingError("Install google-generativeai to use Gemini embeddings.") from exc

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model if model.startswith("models/") else f"models/{model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, task_type="retrieval_document") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, task_type="retrieval_query")

    def _embed(self, text: str, task_type: str) -> list[float]:
        try:
            result = self._genai.embed_content(
                model=self._model,
                content=text,
                task_type=task_type,
                request_options={"timeout": 30},
            )
        except Exception as exc:  # pragma: no cover - SDK/network details vary.
            raise EmbeddingError(
                "Gemini embedding request failed. "
                f"Model: {self._model}. Cause: {type(exc).__name__}: {exc}"
            ) from exc

        embedding = result.get("embedding")
        if not embedding:
            raise EmbeddingError("Gemini returned an empty embedding.")
        return _normalize([float(value) for value in embedding])


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embeddings for offline demos and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        return _normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length.")
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]
