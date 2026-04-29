"""Streamlit UI for the Document Intelligence RAG backend."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="RAG for Document Intelligence", page_icon=":page_facing_up:", layout="wide")


def api_url() -> str:
    """Read Streamlit secrets only when present so local testing works without secrets.toml."""
    local_secrets = Path(".streamlit/secrets.toml")
    home_secrets = Path.home() / ".streamlit" / "secrets.toml"
    if not local_secrets.exists() and not home_secrets.exists():
        return "http://localhost:8000"
    return st.secrets.get("API_URL", "http://localhost:8000").rstrip("/")


def keep_backend_awake() -> None:
    """Ping Railway's health route from the UI because the free tier may sleep after inactivity."""
    try:
        requests.get(f"{api_url()}/health", timeout=5)
    except requests.RequestException:
        pass


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

keep_backend_awake()

st.title("RAG for Document Intelligence")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    force = st.checkbox("Force reingest")
    if st.button("Ingest", disabled=uploaded is None):
        with st.spinner("Embedding and storing chunks..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            data = {"force_reingest": str(force).lower()}
            response = requests.post(f"{api_url()}/ingest", files=files, data=data, timeout=180)
        if response.ok:
            st.success(response.json())
        else:
            st.error(response.text)

    try:
        docs = requests.get(f"{api_url()}/documents", timeout=20).json().get("documents", [])
    except requests.RequestException:
        docs = []
    selected_doc = st.selectbox("Filename filter", ["All documents"] + docs)
    retrieval_mode = st.radio("Retrieval", ["hybrid", "dense"], horizontal=True)
    if st.button("Clear chat"):
        requests.delete(f"{api_url()}/session/{st.session_state.session_id}", timeout=10)
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "retrieval_mode": retrieval_mode,
        "filename_filter": None if selected_doc == "All documents" else selected_doc,
    }
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            response = requests.post(f"{api_url()}/ask", json=payload, timeout=120)
        if response.ok:
            data = response.json()
            st.markdown(data["answer"])
            with st.expander("Citations"):
                for citation in data["citations"]:
                    st.write(citation)
            st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
        else:
            st.error(response.text)

with st.expander("Evaluate"):
    run_name = st.text_input("Run name", value=f"streamlit-{int(time.time())}")
    if st.button("Run RAGAS evaluation"):
        with st.spinner("Running evaluation..."):
            response = requests.post(f"{api_url()}/evaluate", json={"run_name": run_name}, timeout=180)
        st.json(response.json() if response.ok else {"error": response.text})
