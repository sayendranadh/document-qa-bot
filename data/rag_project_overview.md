# RAG Project Overview

Retrieval-Augmented Generation helps a language model answer questions from private documents by retrieving relevant context before generation.

RAG reduces hallucination because the model is not asked to rely only on its training memory. Instead, it receives document chunks that were retrieved from the user's uploaded files and must answer from that grounded context.

The system in this project performs these steps:

1. Extract text from uploaded files.
2. Split text into overlapping chunks.
3. Generate vector embeddings for each chunk.
4. Store the vectors in ChromaDB.
5. Embed the user's query with the same embedding model.
6. Retrieve the top-k most similar chunks.
7. Build a strict prompt that tells the model to answer only from retrieved context.
8. Return a grounded answer with citations.

This approach is useful for internship evaluation because it demonstrates practical knowledge of LLM limitations, vector search, prompt engineering, and deployment.
