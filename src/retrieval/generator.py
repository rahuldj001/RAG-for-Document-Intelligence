"""Groq-backed answer generation with simple Python conversation memory."""

from __future__ import annotations

import os
from collections import defaultdict

from groq import Groq

from src.config import require_env
from src.retrieval.retriever import RetrievedChunk


CONVERSATIONS: dict[str, list[dict[str, str]]] = defaultdict(list)


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format citations into the prompt so the model can ground answers without a retrieval framework."""
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        citation = f"[{index}] {chunk.filename}, page {chunk.page}, chunk {chunk.chunk_index}"
        blocks.append(f"{citation}\n{chunk.content}")
    return "\n\n".join(blocks)


def answer_question(question: str, chunks: list[RetrievedChunk], session_id: str) -> dict:
    """Use Groq directly so generation follows the no-OpenAI and no-LangChain-generation constraints."""
    client = Groq(api_key=require_env("GROQ_API_KEY"))
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    history = CONVERSATIONS[session_id][-6:]
    context = build_context(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using only the provided document context. "
                "Cite sources as [number] after factual claims. If the answer is not in the context, say so."
            ),
        }
    ]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Document context:\n{context}\n\nQuestion: {question}",
        }
    )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=900,
    )
    answer = completion.choices[0].message.content
    CONVERSATIONS[session_id].extend(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    )
    return {
        "answer": answer,
        "citations": [
            {
                "index": i + 1,
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
                "retrieval_method": chunk.retrieval_method,
            }
            for i, chunk in enumerate(chunks)
        ],
    }


def clear_session(session_id: str) -> bool:
    """Clear a plain list-backed memory entry because the app intentionally avoids LangChain memory classes."""
    existed = session_id in CONVERSATIONS
    CONVERSATIONS.pop(session_id, None)
    return existed
