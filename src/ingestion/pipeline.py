"""PDF ingestion for the Document Intelligence pipeline."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

from src.config import require_env


def get_supabase_client() -> Client:
    """Create the Supabase service client here so ingestion can use direct table APIs without an ORM."""
    url = require_env("SUPABASE_URL")
    key = require_env("SUPABASE_SERVICE_KEY")
    return create_client(url, key)


def get_embedding_model() -> SentenceTransformer:
    """Load the local HuggingFace model once per call path so embeddings stay free and avoid external APIs."""
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def md5_file(path: str | Path) -> str:
    """Hash the raw PDF bytes so duplicate detection is based on file identity, not filename drift."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_to_pg(value: list[float]) -> str:
    """Serialize vectors explicitly because PostgREST accepts pgvector text reliably across supabase-py versions."""
    return "[" + ",".join(f"{x:.8f}" for x in value) + "]"


def already_ingested(client: Client, file_hash: str) -> bool:
    """Check metadata by hash before doing expensive PDF parsing and embedding work."""
    response = (
        client.table("documents")
        .select("id")
        .contains("metadata", {"file_hash": file_hash})
        .limit(1)
        .execute()
    )
    return bool(response.data)


def make_splitter(model: SentenceTransformer) -> RecursiveCharacterTextSplitter:
    """Use LangChain only for splitting, with the embedding tokenizer to make 512 mean model tokens."""
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "51"))
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(model.tokenizer.encode(text, add_special_tokens=False)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def ingest_pdf(
    pdf_path: str | Path,
    filename: str | None = None,
    force_reingest: bool = False,
    client: Client | None = None,
) -> dict[str, Any]:
    """Ingest one PDF through the required loader/splitter path and keep deduplication outside retrieval logic."""
    client = client or get_supabase_client()
    model = get_embedding_model()
    pdf_path = Path(pdf_path)
    filename = filename or pdf_path.name
    file_hash = md5_file(pdf_path)

    if not force_reingest and already_ingested(client, file_hash):
        return {"status": "skipped", "filename": filename, "file_hash": file_hash, "chunks": 0}

    if force_reingest:
        client.table("documents").delete().contains("metadata", {"file_hash": file_hash}).execute()

    pages = PyPDFLoader(str(pdf_path)).load()
    splitter = make_splitter(model)
    chunks = splitter.split_documents(pages)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist() if texts else []

    rows = []
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        rows.append(
            {
                "content": chunk.page_content,
                "embedding": vector_to_pg(embedding),
                "metadata": {
                    "filename": filename,
                    "page": int(chunk.metadata.get("page", 0)) + 1,
                    "chunk_index": index,
                    "file_hash": file_hash,
                },
            }
        )

    for start in range(0, len(rows), 100):
        client.table("documents").insert(rows[start : start + 100]).execute()

    return {"status": "ingested", "filename": filename, "file_hash": file_hash, "chunks": len(rows)}


def ingest_upload_bytes(
    pdf_bytes: bytes,
    filename: str,
    force_reingest: bool = False,
) -> dict[str, Any]:
    """Bridge FastAPI uploads to the file-based loader because PyPDFLoader expects a filesystem path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
        handle.write(pdf_bytes)
        temp_path = handle.name
    try:
        return ingest_pdf(temp_path, filename=filename, force_reingest=force_reingest)
    finally:
        Path(temp_path).unlink(missing_ok=True)
