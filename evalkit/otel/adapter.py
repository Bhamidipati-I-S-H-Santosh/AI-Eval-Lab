"""Evaluate recorded traces as if they were a live system.

config:
  system:
    type: otel_replay
    traces: ["traces/*.json"]        # files/globs: OTLP/JSON, JSON-lines, or .pb
    # or traces stored by evalkit's receiver (python -m evalkit serve):
    storage_prefix: otel/            # read from the configured storage backend
    match: input                     # input (default) | trace_id (case.metadata.trace_id)
  dataset: {path: cases.jsonl}       # questions (+ expected answers) to look up

Each test case's question is matched to the trace whose agent input is the
same question (case- and whitespace-insensitive); the newest trace wins.
"""
from __future__ import annotations

import re

from ..adapters.base import AgentAdapter
from ..errors import ConfigError
from ..plugins import HealthStatus
from ..types import AgentResult
from .normalize import NormalizedTrace, load_traces, normalize_spans, parse_otlp


def _key(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class OtelReplayAgent(AgentAdapter):
    def __init__(self, traces: str | list[str] | None = None, storage: dict | None = None,
                 storage_prefix: str = "otel/", spans: list | None = None, **_):
        all_spans = list(spans or [])
        if traces:
            all_spans += load_traces(traces)
        if storage:
            from ..storage import build_storage
            st = build_storage(storage)
            for k in st.list(storage_prefix):
                text = st.read_text(k)
                if text:
                    all_spans += parse_otlp(text)
        if not all_spans:
            raise ConfigError("otel_replay found no spans (check 'traces' globs or 'storage')")
        self.traces: list[NormalizedTrace] = sorted(normalize_spans(all_spans), key=lambda t: t.start_ns or 0)
        self.by_input: dict[str, NormalizedTrace] = {}
        for t in self.traces:                       # newest last -> wins
            if t.question:
                self.by_input[_key(t.question)] = t
        self.by_id = {t.trace_id: t for t in self.traces}

    def run(self, input: str) -> AgentResult:
        t = self.by_input.get(_key(input)) or self.by_id.get(input.strip())
        if t is None:
            return AgentResult(answer="", error="no recorded trace for this input")
        r = t.result
        return AgentResult(answer=r.answer, trace=list(r.trace), retrieved=list(r.retrieved),
                           prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                           latency_s=r.latency_s, error=r.error, status=r.status, raw=r.raw)

    def health_check(self) -> HealthStatus:
        warn = sum(1 for t in self.traces if t.warnings)
        return HealthStatus(True, f"{len(self.traces)} trace(s) loaded, {len(self.by_input)} with an input"
                                  + (f", {warn} with warnings" if warn else ""))
