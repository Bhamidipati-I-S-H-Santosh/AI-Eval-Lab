"""The standard formats every adapter speaks. The evaluator only ever sees these."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolCall:
    tool: str
    args: dict = field(default_factory=dict)
    result: str = ""


@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """What a system returned for one input. Only `answer` is required;
    every optional field that an adapter fills in unlocks more metrics."""
    answer: str
    trace: list[ToolCall] = field(default_factory=list)
    retrieved: list[RetrievedDoc] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    error: str | None = None
    status: str | None = None  # optional self-declared: "answered" | "partial" | "unanswerable"
    raw: Any = None  # the untouched response, for debugging

    @property
    def tools_used(self) -> list[str]:
        return [t.tool for t in self.trace]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw"] = None if self.raw is None else str(self.raw)[:2000]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AgentResult":
        d = dict(d)
        d["trace"] = [ToolCall(**t) for t in d.get("trace") or []]
        d["retrieved"] = [RetrievedDoc(**r) for r in d.get("retrieved") or []]
        return cls(**d)


@dataclass
class TestCase:
    """A generic test case. Every field except id and input is optional; a
    metric that needs a missing field simply returns None and is skipped."""
    id: str
    input: str
    expected_output: str | None = None        # reference answer for the judge
    expected_contains: list[str] = field(default_factory=list)  # exact strings that must appear
    expected_number: float | None = None
    expected_sources: list[str] = field(default_factory=list)   # doc ids that should be retrieved
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    forbidden_text: list[str] = field(default_factory=list)
    unanswerable: bool = False
    tags: list[str] = field(default_factory=list)
    attack: dict | None = None                # set for injection cases: {"id", "canaries", ...}
    turns: list[str] = field(default_factory=list)   # multi-turn: user messages in order (input = last)
    identity: str | None = None               # identity-aware tests: run as this test user
    metadata: dict = field(default_factory=dict)

    __test__ = False  # stop pytest from collecting this class

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        base = {k: v for k, v in d.items() if k in known}
        if "question" in extra and "input" not in base:  # accept a common alias
            base["input"] = extra.pop("question")
        if base.get("turns") and not base.get("input"):
            base["input"] = base["turns"][-1]
        tc = cls(**base)
        tc.metadata.update(extra)
        return tc

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {}, False)}
