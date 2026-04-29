"""RAGAS evaluation runner with Groq/local-embedding fallbacks."""

from __future__ import annotations

import os
import statistics
from datetime import datetime, timezone
from typing import Any

from groq import Groq
from supabase import Client, create_client

from src.config import require_env
from src.retrieval.generator import answer_question
from src.retrieval.retriever import hybrid_search


def get_supabase_client() -> Client:
    """Use supabase-py directly so evaluation results land beside the documents without another store."""
    return create_client(require_env("SUPABASE_URL"), require_env("SUPABASE_SERVICE_KEY"))


def default_eval_questions() -> list[dict[str, str]]:
    """Provide a tiny smoke-eval set so deployments can validate RAG before a curated dataset exists."""
    return [
        {"question": "Summarize the main topic of the uploaded documents.", "ground_truth": ""},
        {"question": "What are the most important facts mentioned in the documents?", "ground_truth": ""},
    ]


def groq_judge(question: str, answer: str, contexts: list[str], ground_truth: str = "") -> dict[str, float]:
    """Ask Groq for RAGAS-style faithfulness and relevance when full RAGAS wrappers are unavailable."""
    client = Groq(api_key=require_env("GROQ_API_KEY"))
    prompt = (
        "Score this RAG answer as JSON with numeric keys faithfulness, answer_relevancy, context_precision, "
        "and context_recall from 0 to 1. Use context_recall=1 when no ground truth is provided and the answer "
        "is supported by context.\n\n"
        f"Question: {question}\nGround truth: {ground_truth}\nAnswer: {answer}\nContexts:\n" + "\n---\n".join(contexts)
    )
    completion = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=250,
        response_format={"type": "json_object"},
    )
    import json

    data = json.loads(completion.choices[0].message.content)
    return {key: float(data.get(key, 0.0)) for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}


def average_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average metric rows so each eval run stores one compact snapshot in Supabase."""
    if not rows:
        return {}
    return {key: statistics.mean(row[key] for row in rows) for key in rows[0].keys()}


def previous_scores(client: Client) -> dict[str, float]:
    """Read the last run so regressions are visible without needing a separate experiment tracker."""
    response = (
        client.table("eval_results")
        .select("scores")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return {}
    return response.data[0].get("scores") or {}


def compute_deltas(current: dict[str, float], previous: dict[str, float]) -> dict[str, float]:
    """Store score movement with the run because free-tier observability should be simple to inspect."""
    return {key: current.get(key, 0.0) - previous.get(key, 0.0) for key in current}


def run_ragas_eval(samples: list[dict[str, str]] | None = None, run_name: str | None = None) -> dict[str, Any]:
    """Run a lightweight RAGAS-compatible evaluation loop and compare it with the previous Supabase run."""
    client = get_supabase_client()
    samples = samples or default_eval_questions()
    metric_rows = []
    details = []

    for sample in samples:
        question = sample["question"]
        chunks = hybrid_search(question)
        generated = answer_question(question, chunks, session_id="__eval__")
        contexts = [chunk.content for chunk in chunks]
        scores = groq_judge(question, generated["answer"], contexts, sample.get("ground_truth", ""))
        metric_rows.append(scores)
        details.append({"question": question, "scores": scores, "citations": generated["citations"]})

    scores = average_scores(metric_rows)
    prior = previous_scores(client)
    deltas = compute_deltas(scores, prior)
    run_name = run_name or f"ragas-{datetime.now(timezone.utc).isoformat()}"
    client.table("eval_results").insert(
        {"run_name": run_name, "scores": scores, "deltas": deltas, "sample_count": len(samples)}
    ).execute()
    return {"run_name": run_name, "scores": scores, "deltas": deltas, "details": details}
