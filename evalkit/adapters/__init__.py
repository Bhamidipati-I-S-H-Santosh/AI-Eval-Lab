"""Adapter registry: config 'type' -> class. Register your own with
@register_agent("name") / @register_retriever("name"), or use type: custom
with  class: my_module:MyAdapter."""
from __future__ import annotations

from ..utils import expand_env, import_object
from .base import AgentAdapter, RetrieverAdapter, RetrieverAsAgent

AGENTS: dict[str, str] = {
    "python": "evalkit.adapters.python_agent:PythonAgentAdapter",
    "http": "evalkit.adapters.http_agent:HttpAgentAdapter",
}
RETRIEVERS: dict[str, str] = {
    "qdrant": "evalkit.adapters.qdrant:QdrantRetrieverAdapter",
    "folder": "evalkit.adapters.folder:FolderRetrieverAdapter",
}


def register_agent(name: str):
    def deco(cls):
        AGENTS[name] = cls
        return cls
    return deco


def register_retriever(name: str):
    def deco(cls):
        RETRIEVERS[name] = cls
        return cls
    return deco


def _build(registry: dict, cfg: dict, kind: str, **extra):
    cfg = expand_env(dict(cfg))
    t = cfg.pop("type")
    if t == "custom":
        cls = import_object(cfg.pop("class"))
    elif t in registry:
        cls = registry[t]
        cls = import_object(cls) if isinstance(cls, str) else cls
    else:
        raise ValueError(f"unknown {kind} type '{t}'. Known: {sorted(registry)} or 'custom'")
    return cls(**cfg, **extra)


def build_agent(cfg: dict) -> AgentAdapter:
    return _build(AGENTS, cfg, "system")


def build_retriever(cfg: dict) -> RetrieverAdapter:
    return _build(RETRIEVERS, cfg, "retriever")


__all__ = ["AgentAdapter", "RetrieverAdapter", "RetrieverAsAgent", "build_agent",
           "build_retriever", "register_agent", "register_retriever"]
