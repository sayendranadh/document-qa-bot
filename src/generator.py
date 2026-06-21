from __future__ import annotations

from abc import ABC, abstractmethod

from .models import RetrievedChunk
from .prompts import build_grounded_prompt


class AnswerGenerationError(RuntimeError):
    """Raised when an LLM cannot generate an answer."""


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        raise NotImplementedError


class GeminiAnswerGenerator(AnswerGenerator):
    """LLM-backed answer generator for grounded answers with citations."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - dependency message for runtime.
            raise AnswerGenerationError("Install google-generativeai to use Gemini answers.") from exc

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        prompt = build_grounded_prompt(question, retrieved_chunks)
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_output_tokens": 900,
                },
                request_options={"timeout": 60},
            )
        except Exception as exc:  # pragma: no cover - SDK/network details vary.
            raise AnswerGenerationError(
                "Gemini answer generation failed. "
                f"Cause: {type(exc).__name__}: {exc}"
            ) from exc

        text = getattr(response, "text", None)
        if not text:
            raise AnswerGenerationError("Gemini returned an empty answer.")
        return text.strip()


class ExtractiveAnswerGenerator(AnswerGenerator):
    """Offline fallback that proves retrieval behavior without an API key."""

    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "No relevant document context was found for this question."

        top = retrieved_chunks[:3]
        evidence_lines = [
            f"[{item.rank}] {item.chunk.text[:450].strip()}"
            for item in top
            if item.chunk.text.strip()
        ]
        evidence = "\n\n".join(evidence_lines)

        return (
            "The app is running in offline demo mode, so it cannot synthesize a Gemini answer. "
            "The most relevant retrieved evidence is shown below and can be used to answer the question:\n\n"
            f"{evidence}"
        )
