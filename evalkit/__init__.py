"""evalkit: evaluate any agent or retrieval system.

Plug a system in with an adapter, describe the run in a YAML config, and get
scores for correctness, retrieval, tool use, cost and prompt-injection
resistance.
"""
__version__ = "0.2.0"

from .adapters import (AgentAdapter, RetrieverAdapter, build_agent, build_retriever,
                       register_agent, register_retriever)
from .plugins import REGISTRY, Plugin, register
from .metrics import metric
from .types import AgentResult, RetrievedDoc, TestCase, ToolCall

__all__ = ["AgentAdapter", "RetrieverAdapter", "AgentResult", "RetrievedDoc", "TestCase", "ToolCall",
           "build_agent", "build_retriever", "register_agent", "register_retriever", "metric",
           "REGISTRY", "Plugin", "register", "__version__"]
