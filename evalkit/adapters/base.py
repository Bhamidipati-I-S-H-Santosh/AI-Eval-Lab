"""Adapter interfaces. To connect a new system, subclass one of these."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import AgentResult, RetrievedDoc


class RetrieverAdapter(ABC):
    """Any searchable store: vector DB, search engine, in-memory index."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        ...

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        """Random documents, used to auto-generate test questions.
        Optional: adapters that can't sample just can't generate tests."""
        raise NotImplementedError(f"{type(self).__name__} cannot sample documents")

    def close(self) -> None:
        pass


class AgentAdapter(ABC):
    """Any system that takes an input and returns an answer: an agent, a RAG
    pipeline, a chatbot API."""

    @abstractmethod
    def run(self, input: str) -> AgentResult:
        ...

    def set_retriever(self, retriever: RetrieverAdapter | None) -> None:
        """Swap the system's search backend, which lets the injection wrapper
        work. Optional: systems that can't do this use 'external' attack mode."""
        raise NotImplementedError(f"{type(self).__name__} does not support retriever injection")

    @property
    def supports_retriever_injection(self) -> bool:
        return type(self).set_retriever is not AgentAdapter.set_retriever

    def close(self) -> None:
        pass


class RetrieverAsAgent(AgentAdapter):
    """Evaluate a database on its own (no LLM): the 'answer' is empty and the
    retrieved docs are scored. Useful for comparing search setups cheaply."""

    def __init__(self, retriever: RetrieverAdapter, k: int = 5):
        self.retriever, self.k = retriever, k

    def run(self, input: str) -> AgentResult:
        return AgentResult(answer="", retrieved=self.retriever.search(input, self.k))

    def set_retriever(self, retriever):
        self.retriever = retriever
