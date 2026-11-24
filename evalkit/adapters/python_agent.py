"""Connect any Python function or object as the system under test.

The target can return a plain string, a dict in AgentResult shape, or an
AgentResult. Point at it with 'module:function' in the config.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from ..types import AgentResult, RetrievedDoc, ToolCall
from ..utils import import_object
from .base import AgentAdapter


def to_agent_result(out: Any) -> AgentResult:
    if isinstance(out, AgentResult):
        return out
    if isinstance(out, str):
        return AgentResult(answer=out)
    if isinstance(out, dict):
        return AgentResult(
            answer=str(out.get("answer", "")),
            trace=[t if isinstance(t, ToolCall) else ToolCall(
                tool=t.get("tool") or t.get("name", ""), args=t.get("args") or t.get("arguments") or {},
                result=str(t.get("result", ""))) for t in out.get("trace") or []],
            retrieved=[r if isinstance(r, RetrievedDoc) else RetrievedDoc(
                id=str(r.get("id", "")), text=r.get("text", ""), score=r.get("score"),
                metadata=r.get("metadata") or {}) for r in out.get("retrieved") or []],
            prompt_tokens=out.get("prompt_tokens"), completion_tokens=out.get("completion_tokens"),
            raw=out)
    raise TypeError(f"can't convert {type(out).__name__} to AgentResult")


class PythonAgentAdapter(AgentAdapter):
    """
    config:
      type: python
      target: my_package.my_agent:answer      # callable(input) -> str | dict | AgentResult
      retriever_hook: my_package.my_agent:set_search   # optional, enables injection tests
    """

    def __init__(self, target: str | Callable, retriever_hook: str | Callable | None = None, **_):
        self.fn = import_object(target) if isinstance(target, str) else target
        self.hook = import_object(retriever_hook) if isinstance(retriever_hook, str) else retriever_hook

    @property
    def supports_retriever_injection(self) -> bool:
        return self.hook is not None

    def set_retriever(self, retriever):
        if self.hook is None:
            raise NotImplementedError("set 'retriever_hook' in the config to enable injection tests")
        self.hook(retriever)

    def run(self, input: str) -> AgentResult:
        t0 = time.time()
        result = to_agent_result(self.fn(input))
        if result.latency_s is None:
            result.latency_s = round(time.time() - t0, 3)
        return result
