"""OpenTelemetry ingestion: evaluate any agent that emits traces.

Any framework or cloud that exports OTLP traces (JSON or protobuf) using the
OpenTelemetry GenAI semantic conventions, OpenInference (LlamaIndex,
LangChain, Phoenix/Arize instrumentations) or OpenLLMetry (Traceloop) can be
evaluated without writing an adapter:

  1. Point the agent's OTLP exporter at evalkit's receiver
     (python -m evalkit serve  ->  POST /v1/traces), or export traces to files.
  2. Evaluate the recorded traces:
       system: {type: otel_replay, traces: "traces/*.json"}
     or turn them into a dataset:  python -m evalkit otel import traces/*.json --out cases.jsonl

Each trace is normalized into the same AgentResult every adapter produces:
question, answer, tool calls (with arguments and results), retrieved
documents, token usage, latency, errors, model, and the span tree.
"""
from .normalize import NormalizedTrace, load_traces, normalize_spans, parse_otlp

__all__ = ["NormalizedTrace", "load_traces", "normalize_spans", "parse_otlp"]
