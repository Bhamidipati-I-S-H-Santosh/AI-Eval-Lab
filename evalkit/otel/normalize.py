"""Parse OTLP traces and normalize GenAI spans into evalkit's result format.

Supported conventions (attribute names, old and new):
  OpenTelemetry GenAI   gen_ai.operation.name (invoke_agent | chat | text_completion |
                        generate_content | execute_tool | embeddings), gen_ai.usage.input_tokens /
                        output_tokens (and the older prompt_tokens / completion_tokens),
                        gen_ai.request.model, gen_ai.response.model, gen_ai.agent.name,
                        gen_ai.tool.name, gen_ai.tool.call.arguments / .result,
                        gen_ai.input.messages / gen_ai.output.messages (JSON),
                        events gen_ai.user.message / gen_ai.choice / gen_ai.tool.message
  OpenInference         openinference.span.kind (AGENT | CHAIN | LLM | TOOL | RETRIEVER),
                        input.value / output.value, llm.token_count.prompt / completion,
                        llm.model_name, tool.name / tool.parameters,
                        retrieval.documents.N.document.id / content / score / metadata
  OpenLLMetry           gen_ai.prompt.N.content / gen_ai.completion.N.content,
                        llm.usage.prompt_tokens / completion_tokens,
                        traceloop.entity.input / traceloop.entity.output, traceloop.span.kind

Input formats: OTLP/JSON (ExportTraceServiceRequest, as written by the OTLP
file exporter or sent to /v1/traces), OTLP/protobuf bytes (needs the 'otel'
extra), or a JSON list / JSON-lines of already-flat spans.
"""
from __future__ import annotations

import base64
import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import AgentResult, RetrievedDoc, ToolCall

# ---------- OTLP parsing ----------


def _any_value(v: dict | None) -> Any:
    """OTLP AnyValue -> Python."""
    if not isinstance(v, dict):
        return v
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return v["intValue"]
    if "arrayValue" in v:
        return [_any_value(x) for x in (v["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _any_value(kv.get("value")) for kv in (v["kvlistValue"] or {}).get("values", [])}
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _attrs(items) -> dict:
    if isinstance(items, dict):        # already flat
        return dict(items)
    return {a["key"]: _any_value(a.get("value")) for a in items or [] if isinstance(a, dict) and "key" in a}


def _hex_id(v) -> str:
    """OTLP/JSON ids are hex strings; some exporters emit base64."""
    if not v:
        return ""
    s = str(v)
    if len(s) in (16, 32) and re.fullmatch(r"[0-9a-fA-F]+", s):     # OTLP/JSON: hex trace/span ids
        return s.lower()
    if len(s) in (12, 24) and re.fullmatch(r"[A-Za-z0-9+/]+=*", s):  # protobuf JSON mapping: base64
        try:
            return base64.b64decode(s).hex()
        except Exception:
            pass
    return s.lower() if re.fullmatch(r"[0-9a-fA-F]+", s) else s


def _ns(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str
    name: str
    start_ns: int | None
    end_ns: int | None
    attributes: dict
    events: list[dict] = field(default_factory=list)
    status_code: str = "UNSET"
    status_message: str = ""
    resource: dict = field(default_factory=dict)
    kind: str = ""

    @property
    def duration_s(self) -> float | None:
        if self.start_ns is None or self.end_ns is None:
            return None
        return max(0.0, (self.end_ns - self.start_ns) / 1e9)


_STATUS = {0: "UNSET", 1: "OK", 2: "ERROR", "STATUS_CODE_UNSET": "UNSET", "STATUS_CODE_OK": "OK",
           "STATUS_CODE_ERROR": "ERROR"}


def _span_from_otlp(s: dict, resource: dict) -> Span:
    st = s.get("status") or {}
    code = st.get("code", 0)
    return Span(
        trace_id=_hex_id(s.get("traceId") or s.get("trace_id")),
        span_id=_hex_id(s.get("spanId") or s.get("span_id")),
        parent_id=_hex_id(s.get("parentSpanId") or s.get("parent_span_id") or s.get("parent_id")),
        name=s.get("name", ""),
        start_ns=_ns(s.get("startTimeUnixNano") or s.get("start_time_unix_nano")),
        end_ns=_ns(s.get("endTimeUnixNano") or s.get("end_time_unix_nano")),
        attributes=_attrs(s.get("attributes")),
        events=[{"name": e.get("name", ""), "attributes": _attrs(e.get("attributes")),
                 "time_ns": _ns(e.get("timeUnixNano"))} for e in s.get("events") or []],
        status_code=_STATUS.get(code, str(code)),
        status_message=st.get("message", ""),
        resource=resource,
        kind=str(s.get("kind", "")),
    )


def parse_otlp(payload: dict | list | bytes | str) -> list[Span]:
    """Spans from an OTLP/JSON request, OTLP/protobuf bytes, or flat spans."""
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            payload = _protobuf_to_dict(bytes(payload))
    if isinstance(payload, str):
        payload = json.loads(payload)
    spans: list[Span] = []
    if isinstance(payload, dict) and ("resourceSpans" in payload or "resource_spans" in payload):
        for rs in payload.get("resourceSpans") or payload.get("resource_spans") or []:
            resource = _attrs((rs.get("resource") or {}).get("attributes"))
            for ss in rs.get("scopeSpans") or rs.get("scope_spans") or rs.get("instrumentationLibrarySpans") or []:
                for s in ss.get("spans") or []:
                    spans.append(_span_from_otlp(s, resource))
    elif isinstance(payload, list):
        for s in payload:
            spans += parse_otlp(s) if isinstance(s, dict) and "resourceSpans" in s else \
                [_span_from_otlp(s, _attrs(s.get("resource") or {}))]
    elif isinstance(payload, dict) and ("spans" in payload):
        spans = [_span_from_otlp(s, {}) for s in payload["spans"]]
    elif isinstance(payload, dict) and ("traceId" in payload or "trace_id" in payload):
        spans = [_span_from_otlp(payload, {})]
    else:
        raise ValueError("not an OTLP trace payload (expected resourceSpans, a span list, or spans)")
    return [s for s in spans if s.trace_id and s.span_id]


def _protobuf_to_dict(data: bytes) -> dict:
    try:
        from google.protobuf.json_format import MessageToDict
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    except ImportError as e:
        raise ValueError("OTLP/protobuf needs: pip install 'evalkit[otel]' (or send OTLP/JSON)") from e
    req = ExportTraceServiceRequest()
    req.ParseFromString(data)
    d = MessageToDict(req)
    # MessageToDict base64-encodes ids; _hex_id handles that.
    return d


def load_traces(paths: str | list[str]) -> list[Span]:
    """Load spans from files/globs: .json (OTLP request or span list), .jsonl, .pb."""
    if isinstance(paths, str):
        paths = [paths]
    files: list[str] = []
    for p in paths:
        files += sorted(glob.glob(p, recursive=True)) or ([p] if Path(p).exists() else [])
    spans: list[Span] = []
    for f in files:
        path = Path(f)
        if path.suffix in (".pb", ".bin", ".protobuf"):
            spans += parse_otlp(path.read_bytes())
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        try:
            spans += parse_otlp(json.loads(text))
        except ValueError:
            for line in text.splitlines():       # JSON lines (one request or span per line)
                if line.strip():
                    spans += parse_otlp(json.loads(line))
    return spans


# ---------- normalization ----------
def _first(attrs: dict, *keys, default=None):
    for k in keys:
        if k in attrs and attrs[k] not in (None, ""):
            return attrs[k]
    return default


def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _message_text(messages, role: str | None = None, last: bool = True) -> str | None:
    """Text from GenAI message JSON (new 'parts' format or {role, content})."""
    messages = _json(messages)
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        return messages if isinstance(messages, str) else None
    picked = [m for m in messages if isinstance(m, dict) and (role is None or m.get("role") == role)]
    if not picked:
        return None
    m = picked[-1] if last else picked[0]
    if isinstance(m.get("content"), str):
        return m["content"]
    parts = m.get("parts") or m.get("content") or []
    texts = [p.get("content") or p.get("text") for p in parts if isinstance(p, dict)
             and p.get("type", "text") == "text"]
    return "\n".join(t for t in texts if t) or None


def _op(span: Span) -> str:
    a = span.attributes
    op = str(_first(a, "gen_ai.operation.name", default="") or "").lower()
    oi = str(_first(a, "openinference.span.kind", "traceloop.span.kind", default="") or "").upper()
    if op in ("invoke_agent", "create_agent") or oi in ("AGENT", "WORKFLOW"):
        return "agent"
    if op == "execute_tool" or oi == "TOOL" or "gen_ai.tool.name" in a or "tool.name" in a:
        return "tool"
    if oi == "RETRIEVER" or op == "retrieval" or any(k.startswith("retrieval.documents.") for k in a):
        return "retriever"
    if op in ("chat", "text_completion", "generate_content", "completion") or oi == "LLM" \
            or "gen_ai.usage.input_tokens" in a or "gen_ai.usage.prompt_tokens" in a \
            or "llm.token_count.prompt" in a or "llm.usage.prompt_tokens" in a:
        return "llm"
    if op == "embeddings" or oi == "EMBEDDING":
        return "embedding"
    if oi == "CHAIN":
        return "chain"
    return "other"


def _span_input(span: Span) -> str | None:
    a = span.attributes
    v = _first(a, "input.value", "traceloop.entity.input", "gen_ai.prompt")
    if v is None:
        v = _message_text(a.get("gen_ai.input.messages"), role="user")
    if v is None:
        prompts = sorted((k for k in a if re.fullmatch(r"gen_ai\.prompt\.\d+\.content", k)),
                         key=lambda k: int(k.split(".")[2]))
        users = [k for k in prompts if a.get(k.replace(".content", ".role"), "user") == "user"]
        if users:
            v = a[users[-1]]
    if v is None:
        for e in span.events:
            if e["name"] == "gen_ai.user.message":
                v = _first(e["attributes"], "content", "gen_ai.event.content", "body") or v
    return _unwrap_io(v)


def _span_output(span: Span) -> str | None:
    a = span.attributes
    v = _first(a, "output.value", "traceloop.entity.output", "gen_ai.completion")
    if v is None:
        v = _message_text(a.get("gen_ai.output.messages"), role="assistant")
    if v is None:
        comps = sorted((k for k in a if re.fullmatch(r"gen_ai\.completion\.\d+\.content", k)),
                       key=lambda k: int(k.split(".")[2]))
        if comps:
            v = a[comps[-1]]
    if v is None:
        for e in span.events:
            if e["name"] == "gen_ai.choice":
                msg = _json(_first(e["attributes"], "message", "gen_ai.event.content", "body"))
                v = (msg.get("content") if isinstance(msg, dict) else msg) or v
    return _unwrap_io(v)


def _unwrap_io(v):
    """input.value / entity.input often hold JSON like {"input": "..."} or {"messages": [...]}."""
    v = _json(v)
    if isinstance(v, str) or v is None:
        return v
    if isinstance(v, dict):
        for k in ("input", "question", "query", "output", "answer", "result", "content", "text"):
            if isinstance(v.get(k), str):
                return v[k]
        if "messages" in v:
            return _message_text(v["messages"]) or json.dumps(v)[:2000]
        return json.dumps(v, ensure_ascii=False)[:2000]
    if isinstance(v, list):
        return _message_text(v) or json.dumps(v, ensure_ascii=False)[:2000]
    return str(v)


def _tokens(span: Span) -> tuple[int | None, int | None]:
    a = span.attributes
    pt = _first(a, "gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.token_count.prompt",
                "llm.usage.prompt_tokens")
    ct = _first(a, "gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens",
                "llm.token_count.completion", "llm.usage.completion_tokens")
    return (int(pt) if pt is not None else None), (int(ct) if ct is not None else None)


def _retrieved_docs(span: Span) -> list[RetrievedDoc]:
    a = span.attributes
    idx = sorted({int(m.group(1)) for k in a if (m := re.match(r"retrieval\.documents\.(\d+)\.", k))})
    docs = []
    for i in idx:
        p = f"retrieval.documents.{i}.document."
        meta = _json(a.get(p + "metadata")) or {}
        meta = meta if isinstance(meta, dict) else {"metadata": meta}
        did = str(_first(a, p + "id", default="") or meta.get("source") or meta.get("file_name") or f"doc{i}")
        meta.setdefault("source", meta.get("file_name") or did)
        score = a.get(p + "score")
        docs.append(RetrievedDoc(id=did, text=str(a.get(p + "content") or ""),
                                 score=float(score) if score is not None else None, metadata=meta))
    return docs


@dataclass
class NormalizedTrace:
    trace_id: str
    question: str | None
    result: AgentResult
    agent_name: str | None = None
    model: str | None = None
    service: str | None = None
    service_version: str | None = None
    start_ns: int | None = None
    warnings: list[str] = field(default_factory=list)

    def to_case(self) -> dict:
        """A test case (no reference answer) for reference-free scoring or review."""
        return {"id": f"trace-{self.trace_id[:16]}", "input": self.question or "",
                "tags": ["from_trace"] + ([self.agent_name] if self.agent_name else []),
                "trace_id": self.trace_id, "recorded_answer": self.result.answer,
                "service_version": self.service_version}


def normalize_spans(spans: list[Span]) -> list[NormalizedTrace]:
    """Group spans by trace and build one result per trace."""
    by_trace: dict[str, list[Span]] = {}
    for s in spans:
        by_trace.setdefault(s.trace_id, []).append(s)
    return [_normalize_trace(tid, ss) for tid, ss in by_trace.items()]


def _normalize_trace(trace_id: str, spans: list[Span]) -> NormalizedTrace:
    warnings: list[str] = []
    ids = {s.span_id for s in spans}
    spans = sorted(spans, key=lambda s: (s.start_ns or 0))
    roots = [s for s in spans if not s.parent_id or s.parent_id not in ids]
    if any(s.parent_id and s.parent_id not in ids for s in spans):
        warnings.append("some parent spans are missing (partial trace); using the earliest span as root")
    agents = [s for s in spans if _op(s) == "agent"]
    root = agents[0] if agents else (roots[0] if roots else spans[0])

    question = _span_input(root)
    answer = _span_output(root)
    llms = [s for s in spans if _op(s) == "llm"]
    if question is None:
        for s in llms:            # first user message the model saw
            question = _span_input(s)
            if question:
                break
    if answer is None and llms:
        answer = _span_output(llms[-1])
    if question is None:
        warnings.append("no input found on the agent/root span or LLM spans")
    if answer is None:
        warnings.append("no output found; the answer is empty")

    trace: list[ToolCall] = []
    retrieved: list[RetrievedDoc] = []
    for s in spans:
        op = _op(s)
        a = s.attributes
        if op == "tool":
            name = str(_first(a, "gen_ai.tool.name", "tool.name", default=s.name))
            args = _json(_first(a, "gen_ai.tool.call.arguments", "tool.parameters", "input.value",
                                "traceloop.entity.input", default={}))
            res = _first(a, "gen_ai.tool.call.result", "output.value", "traceloop.entity.output", default="")
            if not res:
                for e in s.events:
                    if e["name"] in ("gen_ai.tool.message",):
                        res = _first(e["attributes"], "content", "body", default="")
            trace.append(ToolCall(tool=name, args=args if isinstance(args, dict) else {"input": args},
                                  result=res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)))
        elif op == "retriever":
            docs = _retrieved_docs(s)
            retrieved += docs
            trace.append(ToolCall(tool=s.name or "retriever", args={"query": _span_input(s) or ""},
                                  result=f"{len(docs)} document(s)"))
    pt_total = ct_total = 0
    have = False
    for s in llms:
        pt, ct = _tokens(s)
        if pt is not None or ct is not None:
            have = True
            pt_total += pt or 0
            ct_total += ct or 0
    if not have:
        pt, ct = _tokens(root)
        if pt is not None or ct is not None:
            have, pt_total, ct_total = True, pt or 0, ct or 0

    errors = [s for s in spans if s.status_code == "ERROR"]
    error = None
    if root.status_code == "ERROR" or (errors and not answer):
        e = errors[0]
        error = f"span '{e.name}' failed: {e.status_message or e.attributes.get('error.type') or 'error'}"
    start = min((s.start_ns for s in spans if s.start_ns), default=None)
    end = max((s.end_ns for s in spans if s.end_ns), default=None)
    latency = round((end - start) / 1e9, 3) if start and end else root.duration_s

    model = None
    for s in llms or [root]:
        model = _first(s.attributes, "gen_ai.response.model", "gen_ai.request.model", "llm.model_name") or model
    res = root.resource or {}
    tree = [{"span_id": s.span_id, "parent_id": s.parent_id or None, "name": s.name, "op": _op(s),
             "start_ns": s.start_ns, "duration_s": s.duration_s, "status": s.status_code} for s in spans]
    result = AgentResult(
        answer=answer or "", trace=trace, retrieved=retrieved,
        prompt_tokens=pt_total if have else None, completion_tokens=ct_total if have else None,
        latency_s=latency, error=error,
        raw={"trace_id": trace_id, "spans": tree[:200], "warnings": warnings})
    return NormalizedTrace(
        trace_id=trace_id, question=question, result=result,
        agent_name=_first(root.attributes, "gen_ai.agent.name", "agent.name"),
        model=model, service=res.get("service.name"), service_version=res.get("service.version"),
        start_ns=start, warnings=warnings)
