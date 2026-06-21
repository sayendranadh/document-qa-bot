from __future__ import annotations

from .models import RetrievedChunk


def build_grounded_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for item in retrieved_chunks:
        page = item.chunk.metadata.get("page")
        if page:
            citation = f"Source: {item.chunk.document_name}, Page: {page}"
        else:
            citation = f"Source: {item.chunk.document_name}"
        context_blocks.append(
            f"[Chunk {item.rank}]\n"
            f"[{citation}]\n"
            f"Relevance score: {item.score:.3f}\n"
            f'"{item.chunk.text}"'
        )

    context = "\n\n---\n\n".join(context_blocks) or "No retrieved context satisfied the relevance threshold."

    return f"""
You are a precise document Q&A assistant.

Use ONLY the provided context to answer the user's question. If the answer cannot be found in the context, say exactly: "I cannot find the answer in the provided documents." Do not attempt to use your own knowledge to answer.

Mention citations alongside facts using the source metadata, for example: (annual_report.pdf, Page 12). Keep the answer clear, concise, and professional.

Context:
{context}

Question:
{question}

Grounded answer:
""".strip()
