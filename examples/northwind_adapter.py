"""Plugs the Northwind agent (src/) into evalkit.

This file is the whole integration: ~60 lines. Any other Python agent is
connected the same way: a run() that returns a dict, plus an optional hook to
swap the agent's search backend for injection testing.
"""
from __future__ import annotations

import json

from evalkit.adapters.base import RetrieverAdapter
from evalkit.types import RetrievedDoc
from src import tools
from src.agent import run_agent
from src.ingest import Chunk, build_chunks
from src.retrieve import Retriever


def run(question: str) -> dict:
    r = run_agent(question)
    retrieved = []
    for s in r.steps:
        if s.tool == "search_policies":
            try:
                for item in json.loads(s.result):
                    retrieved.append({"id": item["source"], "text": item["text"],
                                      "metadata": {"section": item.get("section")}})
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return {"answer": r.answer, "status": r.status,
            "trace": [{"tool": s.tool, "args": s.args, "result": s.result} for s in r.steps],
            "retrieved": retrieved,
            "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}


class _ChunkShim:
    """Makes any evalkit RetrieverAdapter look like the agent's own retriever."""

    def __init__(self, retriever: RetrieverAdapter):
        self.retriever = retriever

    def search(self, query: str, top_k: int = 3):
        return [Chunk(id=d.id, source=d.metadata.get("source", d.id.split("#")[0]),
                      title=d.metadata.get("title", ""), text=d.text)
                for d in self.retriever.search(query, top_k)]


def set_retriever(retriever: RetrieverAdapter | None) -> None:
    tools.set_retriever(None if retriever is None else _ChunkShim(retriever))


class NorthwindRetriever(RetrieverAdapter):
    """The agent's own search, exposed to evalkit (for retrieval scores,
    test generation, and as the base the injection wrapper poisons)."""

    def __init__(self, method: str = "bm25", **_):
        self.chunks = build_chunks()
        self.r = Retriever(self.chunks, method=method)

    def search(self, query: str, k: int = 5):
        return [RetrievedDoc(id=c.id, text=c.text, metadata={"source": c.source, "title": c.title})
                for c in self.r.search(query, top_k=k)]

    def sample(self, n: int, seed: int = 0):
        import random
        picks = random.Random(seed).sample(self.chunks, min(n, len(self.chunks)))
        return [RetrievedDoc(id=c.id, text=c.text, metadata={"source": c.source, "title": c.title})
                for c in picks]
