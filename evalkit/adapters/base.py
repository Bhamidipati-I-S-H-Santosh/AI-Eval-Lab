"""Adapter interfaces. To connect a new system, subclass one of these."""
from __future__ import annotations

from abc import abstractmethod

from ..plugins import HealthStatus, Plugin, timed_health
from ..types import AgentResult, RetrievedDoc, ToolCall


class RetrieverAdapter(Plugin):
    """Any searchable store: vector DB, search engine, in-memory index."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        ...

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        """Random documents, used to auto-generate test questions.
        Optional: adapters that can't sample just can't generate tests."""
        raise NotImplementedError(f"{type(self).__name__} cannot sample documents")

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"search returned {len(self.search('health check', 1))} doc(s)")

    def close(self) -> None:
        pass


class AgentAdapter(Plugin):
    """Any system that takes an input and returns an answer: an agent, a RAG
    pipeline, a chatbot API."""

    @abstractmethod
    def run(self, input: str) -> AgentResult:
        ...

    def run_conversation(self, turns: list[str]) -> AgentResult:
        """Multi-turn test cases. Systems with native sessions (threads,
        conversation ids) override this. The default replays the conversation
        as a transcript, one call per user turn, so any single-turn system can
        still be scored on the final answer; every turn's answer is kept."""
        history: list[tuple[str, str]] = []
        trace: list[ToolCall] = []
        retrieved: list[RetrievedDoc] = []
        pt = ct = 0
        have_tokens = False
        last = AgentResult(answer="")
        for turn in turns:
            if history:
                convo = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history)
                prompt = f"Conversation so far:\n{convo}\n\nUser: {turn}"
            else:
                prompt = turn
            last = self.run(prompt)
            history.append((turn, last.answer))
            trace += last.trace
            retrieved += last.retrieved
            if last.prompt_tokens is not None or last.completion_tokens is not None:
                have_tokens = True
                pt += last.prompt_tokens or 0
                ct += last.completion_tokens or 0
            if last.error:
                break
        return AgentResult(answer=last.answer, trace=trace, retrieved=retrieved,
                           prompt_tokens=pt if have_tokens else None,
                           completion_tokens=ct if have_tokens else None,
                           error=last.error, status=last.status,
                           raw={"turns": [{"user": u, "assistant": a} for u, a in history]})

    def set_retriever(self, retriever: RetrieverAdapter | None) -> None:
        """Swap the system's search backend, which lets the injection wrapper
        work. Optional: systems that can't do this use 'external' attack mode."""
        raise NotImplementedError(f"{type(self).__name__} does not support retriever injection")

    @property
    def supports_retriever_injection(self) -> bool:
        return type(self).set_retriever is not AgentAdapter.set_retriever

    @property
    def supports_identity(self) -> bool:
        """True if this adapter can switch the end-user identity itself
        (identity-aware tests). Otherwise give each test user its own
        'system' overrides (e.g. its own token) in the identity config."""
        return False

    def set_identity(self, name: str | None) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot switch identity")

    def health_check(self) -> HealthStatus:
        def probe():
            r = self.run("Health check: reply with OK.")
            if r.error:
                raise RuntimeError(r.error)
            return f"answered ({len(r.answer)} chars)"
        return timed_health(probe)

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

    def health_check(self) -> HealthStatus:
        return self.retriever.health_check()
