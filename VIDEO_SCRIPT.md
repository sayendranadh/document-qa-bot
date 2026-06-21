# 3-5 Minute Screen Recording Script

## 0:00-0:30: Project Introduction

Introduce the project as a Document Q&A Bot using Retrieval-Augmented Generation. Explain that the goal is to answer questions from private documents while reducing hallucinations through retrieval and citations.

## 0:30-1:30: Feature Demo

Open the deployed Streamlit app. Upload a sample document, click `Build RAG Index`, ask a question, and open `Retrieved evidence` to show the chunks, similarity scores, source names, and page metadata where available.

## 1:30-2:45: Code Walkthrough

Show the main files:

- `src/document_loader.py`: extracts text and metadata from PDF, DOCX, TXT, and Markdown files.
- `src/chunker.py`: performs recursive character splitting with overlap.
- `src/embeddings.py`: creates Gemini embeddings or offline demo embeddings.
- `src/vector_store.py`: persists and searches vectors in ChromaDB.
- `src/prompts.py`: enforces strict grounding and citation behavior.
- `src/ingest.py` and `src/query.py`: separate indexing from querying.
- `src/main.py`: Streamlit interface.

## 2:45-3:45: Architecture Explanation

Explain the flow: documents are parsed, chunked, embedded, stored in ChromaDB, retrieved by semantic similarity, inserted into a grounded prompt, and then answered by Gemini with citations.

## 3:45-4:30: Why This Is Reliable

Mention that the model is instructed to use only provided context. If the answer is not present, it should say: `I cannot find the answer in the provided documents.` Explain that retrieved evidence is visible so evaluators can verify the answer.

## 4:30-5:00: Closing

Summarize the tech stack: Python, Streamlit, ChromaDB, Gemini, pypdf, python-docx, dotenv, and tqdm. End by showing the GitHub repository and the deployed app URL.
