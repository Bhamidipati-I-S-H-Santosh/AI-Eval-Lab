"""Answer a question from retrieved chunks."""
from __future__ import annotations

import sys

from .ingest import Chunk, build_chunks
from .llm import chat
from .retrieve import Retriever

SYSTEM = (
    "You answer questions about Northwind Labs company policy. "
    "Use ONLY the provided context. If the context does not contain the answer, "
    "reply exactly: I don't know. Keep answers to one or two sentences and "
    "cite the source file in square brackets, e.g. [leave_policy.md]."
)


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.source}] {c.title}\n{c.text}" for c in chunks)


def answer(question: str, retriever: Retriever, top_k: int = 3) -> tuple[str, list[Chunk]]:
    chunks = retriever.search(question, top_k=top_k)
    prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    return chat(prompt, system=SYSTEM), chunks


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many days of annual leave do I get?"
    r = Retriever(build_chunks(), method="bm25")
    text, used = answer(q, r)
    print(text)
    print("\nRetrieved:", [c.id for c in used])
