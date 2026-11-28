"""Adapter registry: config 'type' -> class. All lookups go through the shared
plugin registry (evalkit.plugins), so systems and retrievers from installed
packages (entry points) work here too.

Register your own with @register_agent("name") / @register_retriever("name"),
or use type: custom with  class: my_module:MyAdapter."""
from __future__ import annotations

from ..plugins import REGISTRY
from .base import AgentAdapter, RetrieverAdapter, RetrieverAsAgent


class _CategoryView(dict):
    """Backward-compatible dict view of one registry category (AGENTS, RETRIEVERS)."""

    def __init__(self, category: str):
        super().__init__()
        self.category = category

    def _names(self):
        return [s.name for s in REGISTRY.list(self.category)]

    def __iter__(self):
        return iter(self._names())

    def __contains__(self, name):
        return name in self._names()

    def __len__(self):
        return len(self._names())

    def __getitem__(self, name):
        return REGISTRY.get(self.category, name)

    def __setitem__(self, name, cls):
        REGISTRY.register(self.category, name, cls)

    def keys(self):
        return self._names()


AGENTS = _CategoryView("system")
RETRIEVERS = _CategoryView("retriever")


def register_agent(name: str):
    def deco(cls):
        REGISTRY.register("system", name, cls)
        return cls
    return deco


def register_retriever(name: str):
    def deco(cls):
        REGISTRY.register("retriever", name, cls)
        return cls
    return deco


def build_agent(cfg: dict) -> AgentAdapter:
    return REGISTRY.build("system", cfg)


def build_retriever(cfg: dict) -> RetrieverAdapter:
    return REGISTRY.build("retriever", cfg)


__all__ = ["AgentAdapter", "RetrieverAdapter", "RetrieverAsAgent", "build_agent",
           "build_retriever", "register_agent", "register_retriever", "AGENTS", "RETRIEVERS"]
