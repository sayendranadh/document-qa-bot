from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.config import AppConfig
from src.document_loader import DocumentLoadError, load_uploaded_document
from src.rag_pipeline import RAGPipeline


CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
    }
    .citation {
        border-left: 3px solid #2563eb;
        padding: 0.7rem 0.9rem;
        background: #f8fafc;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.88rem;
    }
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="Document Q&A RAG Bot",
        page_icon=":page_facing_up:",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _initialize_state()

    with st.sidebar:
        config = _render_sidebar()

    _render_header()
    _render_status_metrics(config)
    _render_upload_panel(config)
    _render_chat_panel(config)


def _initialize_state() -> None:
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("document_names", [])
    st.session_state.setdefault("chunk_count", 0)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_indexed_at", None)


def _render_sidebar() -> AppConfig:
    env_config = AppConfig.from_environment()

    st.subheader("Configuration")
    api_key = st.text_input(
        "Gemini API key",
        value=env_config.gemini_api_key or "",
        type="password",
        help="Leave empty to use offline retrieval demo mode.",
    )
    chat_model = st.text_input("Chat model", value=env_config.chat_model)
    embedding_model = st.text_input("Embedding model", value=env_config.embedding_model)

    st.divider()
    chunk_size = st.slider("Chunk size", min_value=400, max_value=2_000, value=1_000, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=400, value=180, step=20)
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=5)
    default_min_score = 0.10 if api_key.strip() else -1.0
    min_score = st.slider(
        "Minimum similarity",
        min_value=-1.0,
        max_value=1.0,
        value=default_min_score,
        step=0.05,
    )

    st.divider()
    if st.button("Clear session", use_container_width=True):
        st.session_state.pipeline = None
        st.session_state.document_names = []
        st.session_state.chunk_count = 0
        st.session_state.messages = []
        st.session_state.last_indexed_at = None
        st.rerun()

    return AppConfig(
        gemini_api_key=api_key.strip() or None,
        chat_model=chat_model.strip() or env_config.chat_model,
        embedding_model=embedding_model.strip() or env_config.embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        min_score=min_score,
        persist_directory=env_config.persist_directory,
        show_progress=True,
    )


def _render_header() -> None:
    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.title("Document Q&A Bot with RAG")
        st.caption("Upload private documents, retrieve the most relevant chunks, and generate grounded answers with citations.")
    with right:
        mode = st.session_state.pipeline.mode if st.session_state.pipeline else "Not indexed"
        st.info(mode)


def _render_status_metrics(config: AppConfig) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Documents", len(st.session_state.document_names))
    col2.metric("Chunks", st.session_state.chunk_count)
    col3.metric("Top K", config.top_k)
    col4.metric("Min Score", f"{config.min_score:.2f}")
    col5.metric("Last Indexed", st.session_state.last_indexed_at or "-")


def _render_upload_panel(config: AppConfig) -> None:
    with st.expander("Document Library", expanded=st.session_state.pipeline is None):
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, or MD files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        build_clicked = st.button(
            "Build RAG Index",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        if build_clicked and uploaded_files:
            with st.spinner("Reading documents, creating chunks, and building semantic index..."):
                try:
                    documents = []
                    for uploaded_file in uploaded_files:
                        documents.extend(load_uploaded_document(uploaded_file, uploaded_file.name))

                    pipeline = RAGPipeline.build(documents, config)
                    st.session_state.pipeline = pipeline
                    st.session_state.document_names = sorted({document.name for document in documents})
                    st.session_state.chunk_count = pipeline.vector_store.count
                    st.session_state.messages = []
                    st.session_state.last_indexed_at = datetime.now().strftime("%H:%M")
                except (DocumentLoadError, ValueError, RuntimeError) as exc:
                    st.error(str(exc))
                else:
                    st.success("RAG index is ready.")
                    st.rerun()

        if st.session_state.document_names:
            st.write("Indexed files")
            for name in st.session_state.document_names:
                st.markdown(f"- `{name}`")


def _render_chat_panel(config: AppConfig) -> None:
    st.subheader("Ask Questions")

    if st.session_state.pipeline is None:
        st.warning("Build an index from uploaded documents before asking questions.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    question = st.chat_input("Ask a question about the uploaded documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating a grounded answer..."):
            try:
                result = st.session_state.pipeline.ask(
                    question,
                    top_k=config.top_k,
                    min_score=config.min_score,
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return

        st.markdown(result.answer)
        _render_sources(result.retrieved_chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.retrieved_chunks,
        }
    )


def _render_sources(retrieved_chunks) -> None:
    with st.expander("Retrieved evidence", expanded=False):
        for item in retrieved_chunks:
            chunk = item.chunk
            st.markdown(
                f"""
<div class="citation">
    <strong>[{item.rank}] {chunk.citation_label}</strong>
    <div class="small-muted">Similarity: {item.score:.3f} | Chunk {chunk.metadata.get("chunk", "-")}</div>
    <p>{_html_escape(chunk.text[:900])}</p>
</div>
""",
                unsafe_allow_html=True,
            )


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
