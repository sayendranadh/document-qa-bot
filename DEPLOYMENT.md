# Deployment Guide

This app is ready for Streamlit Community Cloud or any Python web host that can run Streamlit.

## Local Production Check

Run these commands before deployment:

```powershell
python -m unittest discover -s tests
python -m compileall src tests
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app and select this repository.
4. Set the main file path to `app.py`.
5. Ensure the Python runtime uses `runtime.txt`.
6. Add secrets in the Streamlit app settings:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_PERSIST_DIRECTORY = "db"
```

7. Deploy the app.

## Why Streamlit Works Well Here

Streamlit is simple for evaluators: upload documents, build the index, ask questions, and inspect citations in one browser page. ChromaDB stores vectors locally inside the deployment container, which is enough for a single-user evaluation demo.

## Deployment Checklist

- `requirements.txt` includes all runtime dependencies.
- `runtime.txt` pins Python 3.11 for package compatibility.
- Secrets are not committed.
- `.streamlit/secrets.toml.example` shows the expected secret format.
- Uploaded documents are processed at runtime and are not committed.
- Retrieved sources are visible for trust and explainability.

## Submission Links

After deployment, add the final links to `SUBMISSION_CHECKLIST.md`:

- Public GitHub repository URL
- Streamlit live app URL
- Google Drive, Loom, or YouTube screen recording URL

## Scaling Notes

For a larger production system, replace the single-session upload flow with authenticated users, durable object storage, a managed vector database, background ingestion jobs, and access-control checks before retrieval.
