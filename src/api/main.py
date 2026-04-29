"""FastAPI surface for ingestion, retrieval, generation, and evaluation."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.evaluation.ragas_eval import run_ragas_eval
from src.ingestion.pipeline import get_supabase_client, ingest_upload_bytes
from src.retrieval.generator import answer_question, clear_session
from src.retrieval.retriever import compare_retrievers, dense_search, hybrid_search


app = FastAPI(title="RAG for Document Intelligence")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """Request model kept explicit so the UI can switch retrievers without changing backend code."""
    question: str
    session_id: str
    retrieval_mode: Literal["hybrid", "dense"] = "hybrid"
    filename_filter: str | None = None


class EvalRequest(BaseModel):
    """Evaluation input accepts curated samples while still supporting a zero-config smoke test."""
    run_name: str | None = None
    samples: list[dict[str, str]] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Expose a cheap endpoint for Streamlit keep-alive pings on sleeping Railway apps."""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    force_reingest: bool = Form(False),
) -> dict:
    """Accept PDFs as uploads so ingestion works locally and after Railway deployment."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    return ingest_upload_bytes(await file.read(), file.filename, force_reingest=force_reingest)


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    """Retrieve with direct Supabase calls, then answer with Groq while maintaining plain Python memory."""
    if payload.retrieval_mode == "dense":
        chunks = dense_search(payload.question, filename_filter=payload.filename_filter)
    else:
        chunks = hybrid_search(payload.question, filename_filter=payload.filename_filter)
    result = answer_question(payload.question, chunks, payload.session_id)
    result["chunks"] = [chunk.__dict__ for chunk in chunks]
    return result


@app.get("/documents")
def documents() -> dict[str, list[str]]:
    """List unique filenames from metadata because Supabase free tier should avoid extra catalog tables."""
    rows = get_supabase_client().table("documents").select("metadata").limit(10000).execute().data or []
    filenames = sorted({row.get("metadata", {}).get("filename") for row in rows if row.get("metadata", {}).get("filename")})
    return {"documents": filenames}


@app.get("/compare")
def compare(query: str, filename_filter: str | None = None) -> dict:
    """Expose dense versus hybrid results to make retrieval quality inspectable from the UI or curl."""
    results = compare_retrievers(query, filename_filter=filename_filter)
    return {key: [chunk.__dict__ for chunk in value] for key, value in results.items()}


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    """Clear in-process memory for one chat because this starter intentionally avoids persistent memory."""
    return {"cleared": clear_session(session_id)}


@app.post("/evaluate")
def evaluate(payload: EvalRequest) -> dict:
    """Run the RAGAS-style evaluator and persist scores plus deltas for trend checking."""
    return run_ragas_eval(samples=payload.samples, run_name=payload.run_name)

