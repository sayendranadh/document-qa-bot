# Project Explanation for Internship Evaluation

## One-Minute Summary

This project is a Document Q&A Bot built with Retrieval-Augmented Generation. A user uploads private documents, the app extracts and chunks the text, converts each chunk into embeddings, stores those embeddings in ChromaDB, retrieves the most relevant chunks for a question, and asks Gemini to answer only from that retrieved context. The final answer includes citations so the evaluator can verify the source.

## Problem Solved

Normal LLMs do not know private files unless those files are provided in the prompt. They can also hallucinate when they do not have enough information. This project reduces that risk by grounding each answer in retrieved document chunks.

## Pipeline Walkthrough

1. Document upload
   The user uploads PDF, DOCX, TXT, or Markdown files in the Streamlit UI.

2. Text extraction
   `src/document_loader.py` extracts text using `pypdf` for PDF files and `python-docx` for Word documents. It can process uploaded files, individual file paths, or a full document folder. Each extracted record keeps source metadata such as file name and PDF page number.

3. Chunking
   `src/chunker.py` uses recursive character splitting. It first tries paragraph breaks, then line breaks, sentence-like punctuation, commas, and spaces before falling back to a hard character boundary. Overlap helps preserve meaning when a sentence or idea crosses a chunk boundary.

4. Embedding
   `src/embeddings.py` converts chunks into dense vectors using Gemini `gemini-embedding-001`.

5. Vector storage
   `src/vector_store.py` stores vectors in ChromaDB, a local persistent vector database.

6. Retrieval
   For each question, the query is embedded with the same embedding model used during ingestion. ChromaDB returns the top-k most similar chunks, and the app filters low-score matches before generation.

7. Prompting
   `src/prompts.py` builds a grounded prompt containing only retrieved evidence and strict instructions to cite sources.

8. Generation
   `src/generator.py` sends the prompt to Gemini and returns a concise answer with citations.

## Key Design Decisions

- Streamlit was chosen because it gives evaluators an easy browser-based demo.
- ChromaDB was chosen because it is lightweight, local, persistent, and does not require a separate server.
- Chunk overlap was added to avoid losing context at chunk boundaries.
- Citations are shown because a RAG system should be verifiable, not just fluent.
- Offline demo mode was added so the retrieval path can still be demonstrated without an API key.
- `src/ingest.py` and `src/query.py` are separated so documents can be indexed once and queried many times.

## Questions You Can Answer in Evaluation

### Why not fine-tune the model?

Fine-tuning changes model behavior, but it is expensive and not ideal for frequently changing private documents. RAG is better here because new documents can be indexed without retraining.

### How does the bot reduce hallucination?

The model receives relevant document chunks and is instructed to answer only from that context. If the retrieved context is insufficient, it should say the documents do not contain enough information.

### Why use embeddings?

Embeddings represent meaning numerically. This allows semantic search, so a query can match relevant text even when it uses different wording.

### Why use ChromaDB?

ChromaDB stores embeddings and supports fast vector similarity search locally. It is simple to run for an internship project and does not require a separate database server.

### What are the limitations?

The system depends on quality text extraction, chunk size, embedding quality, and whether the relevant information exists in the uploaded documents. Scanned PDFs may need OCR, which is not included in this version.
