"""Direct Supabase retrieval with dense, BM25, and reciprocal-rank fusion."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

from src.config import require_env


@dataclass
class RetrievedChunk:
    """Small return type so API, UI, and generator agree without depending on LangChain documents."""
    content: str
    filename: str
    page: int
    chunk_index: int
    score: float
    retrieval_method: str


def get_supabase_client() -> Client:
    """Create a plain supabase-py client because retrieval must call RPCs directly instead of an ORM."""
    return create_client(require_env("SUPABASE_URL"), require_env("SUPABASE_SERVICE_KEY"))


def get_embedding_model() -> SentenceTransformer:
    """Keep embeddings local so query-time retrieval remains inside the free HuggingFace setup."""
    return SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))


def tokenize(text: str) -> list[str]:
    """Use a tiny deterministic tokenizer so BM25 has no hidden service dependency."""
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def row_to_chunk(row: dict[str, Any], score: float, method: str) -> RetrievedChunk:
    """Normalize Supabase rows into the project dataclass so fusion and generation stay provider-agnostic."""
    metadata = row.get("metadata") or {}
    return RetrievedChunk(
        content=row["content"],
        filename=metadata.get("filename", ""),
        page=int(metadata.get("page", 0)),
        chunk_index=int(metadata.get("chunk_index", 0)),
        score=float(score),
        retrieval_method=method,
    )


def dense_search(
    question: str,
    top_k: int | None = None,
    filename_filter: str | None = None,
    client: Client | None = None,
    model: SentenceTransformer | None = None,
) -> list[RetrievedChunk]:
    """Call the match_documents RPC directly to keep vector retrieval transparent and LangChain-free."""
    client = client or get_supabase_client()
    model = model or get_embedding_model()
    top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
    query_embedding = model.encode(question, normalize_embeddings=True).tolist()
    response = client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "filename_filter": filename_filter,
        },
    ).execute()
    return [row_to_chunk(row, row.get("similarity", 0.0), "dense") for row in response.data or []]


def fetch_documents_for_bm25(client: Client, filename_filter: str | None = None) -> list[dict[str, Any]]:
    """Fetch document text directly from Supabase because BM25 ranking is intentionally local and inspectable."""
    query = client.table("documents").select("id, content, metadata").limit(10000)
    if filename_filter:
        query = query.contains("metadata", {"filename": filename_filter})
    return query.execute().data or []


def bm25_search(
    question: str,
    top_k: int | None = None,
    filename_filter: str | None = None,
    client: Client | None = None,
) -> list[RetrievedChunk]:
    """Rank lexical matches locally so hybrid search avoids LangChain retrievers and external search services."""
    client = client or get_supabase_client()
    top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
    rows = fetch_documents_for_bm25(client, filename_filter)
    if not rows:
        return []
    corpus = [tokenize(row["content"]) for row in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    return [row_to_chunk(rows[index], float(score), "bm25") for index, score in ranked if score > 0]


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    bm25: list[RetrievedChunk],
    top_k: int | None = None,
    k: int = 60,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[RetrievedChunk]:
    """Fuse ranked lists by stable chunk identity so dense and lexical evidence can reinforce each other."""
    top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
    fused: dict[tuple[str, int, int], RetrievedChunk] = {}
    scores: dict[tuple[str, int, int], float] = {}

    for weight, method, results in ((dense_weight, "hybrid", dense), (bm25_weight, "hybrid", bm25)):
        for rank, chunk in enumerate(results, start=1):
            key = (chunk.filename, chunk.page, chunk.chunk_index)
            fused.setdefault(key, chunk)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank))
            fused[key].retrieval_method = method

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    output = []
    for key, score in ordered:
        chunk = fused[key]
        output.append(
            RetrievedChunk(
                content=chunk.content,
                filename=chunk.filename,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                score=score,
                retrieval_method="hybrid",
            )
        )
    return output


def hybrid_search(
    question: str,
    top_k: int | None = None,
    filename_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Run dense RPC plus local BM25, then RRF them with the requested production weights."""
    client = get_supabase_client()
    model = get_embedding_model()
    top_k = top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
    dense = dense_search(question, top_k=top_k * 2, filename_filter=filename_filter, client=client, model=model)
    bm25 = bm25_search(question, top_k=top_k * 2, filename_filter=filename_filter, client=client)
    return reciprocal_rank_fusion(dense, bm25, top_k=top_k)


def compare_retrievers(question: str, filename_filter: str | None = None) -> dict[str, list[RetrievedChunk]]:
    """Return side-by-side results so users can inspect why hybrid differs from dense-only retrieval."""
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    return {
        "dense": dense_search(question, top_k=top_k, filename_filter=filename_filter),
        "hybrid": hybrid_search(question, top_k=top_k, filename_filter=filename_filter),
    }
