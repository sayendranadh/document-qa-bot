# Submission Checklist

Use this file as the final handoff checklist for internship evaluation.

## Required Deliverables

- GitHub Repository Link: `PASTE_PUBLIC_GITHUB_REPO_URL_HERE`
- Published Project Link: `PASTE_STREAMLIT_OR_HOSTED_APP_URL_HERE`
- Screen Recording Link: `PASTE_GOOGLE_DRIVE_LOOM_OR_YOUTUBE_URL_HERE`

## Before Submitting

- Confirm the repository is public.
- Confirm `.env` and `.streamlit/secrets.toml` are not committed.
- Confirm `requirements.txt` and `runtime.txt` are included.
- Confirm the hosted app opens successfully.
- Upload at least one document and build the RAG index in the hosted app.
- Ask a question and verify that retrieved evidence appears under the answer.
- Record a 3-5 minute walkthrough using `VIDEO_SCRIPT.md`.

## Suggested Demo Question

Upload `sample_documents/internship_rag_project_brief.txt` and ask:

```text
Why does the project use retrieval before generation?
```

The answer should explain that retrieval supplies relevant uploaded-document context before generation, which improves grounding and enables citations.
