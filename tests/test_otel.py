"""OpenTelemetry ingestion. The 'real SDK' tests build spans with the official
opentelemetry-sdk and encode them with the official OTLP exporter's encoder,
so the parser is checked against genuine OTLP protobuf/JSON, not hand-written
fixtures. They skip if the SDK is not installed."""
import json

import pytest

from evalkit.otel import load_traces, normalize_spans, parse_otlp
from evalkit.otel.adapter import OtelReplayAgent
from evalkit.plugins import REGISTRY


def _otlp_json(spans):
    """OTLP/JSON request with the shape the collector and file exporter use."""
    def av(v):
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}
        if isinstance(v, float):
            return {"doubleValue": v}
        return {"stringValue": str(v)}
    out = []
    for s in spans:
        out.append({"traceId": s["trace"], "spanId": s["id"], "parentSpanId": s.get("parent", ""),
                    "name": s["name"], "startTimeUnixNano": str(s["start"]), "endTimeUnixNano": str(s["end"]),
                    "attributes": [{"key": k, "value": av(v)} for k, v in s["attrs"].items()],
                    "events": s.get("events", []), "status": {"code": s.get("status", 1)}})
    return {"resourceSpans": [{"resource": {"attributes": [
        {"key": "service.name", "value": {"stringValue": "hr-bot"}},
        {"key": "service.version", "value": {"stringValue": "1.4.0"}}]},
        "scopeSpans": [{"scope": {"name": "test"}, "spans": out}]}]}


T = "0af7651916cd43dd8448eb211c80319c"


def genai_trace(trace=T, question="How many days of leave?", answer="You get 24 days [leave_policy.md]."):
    return _otlp_json([
        {"trace": trace, "id": "b7ad6b7169203331", "name": "invoke_agent hr-bot", "start": 1_000_000_000,
         "end": 4_500_000_000, "attrs": {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "hr-bot",
                                          "gen_ai.input.messages": json.dumps([{"role": "user", "parts": [{"type": "text", "content": question}]}]),
                                          "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": answer}]}])}},
        {"trace": trace, "id": "00f067aa0ba902b7", "parent": "b7ad6b7169203331", "name": "chat gpt-4.1",
         "start": 1_100_000_000, "end": 2_000_000_000,
         "attrs": {"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4.1",
                   "gen_ai.usage.input_tokens": 120, "gen_ai.usage.output_tokens": 30}},
        {"trace": trace, "id": "11f067aa0ba902b7", "parent": "b7ad6b7169203331", "name": "execute_tool search",
         "start": 2_000_000_000, "end": 2_500_000_000,
         "attrs": {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search_policies",
                   "gen_ai.tool.call.arguments": json.dumps({"query": "annual leave"}),
                   "gen_ai.tool.call.result": "Employees get 24 days."}},
        {"trace": trace, "id": "22f067aa0ba902b7", "parent": "b7ad6b7169203331", "name": "chat gpt-4.1",
         "start": 2_600_000_000, "end": 4_400_000_000,
         "attrs": {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 200, "gen_ai.usage.output_tokens": 25}},
    ])


def test_genai_semconv_trace_is_normalized():
    [t] = normalize_spans(parse_otlp(genai_trace()))
    r = t.result
    assert t.question == "How many days of leave?"
    assert r.answer == "You get 24 days [leave_policy.md]."
    assert [c.tool for c in r.trace] == ["search_policies"] and r.trace[0].args == {"query": "annual leave"}
    assert r.prompt_tokens == 320 and r.completion_tokens == 55
    assert r.latency_s == 3.5 and r.error is None
    assert t.agent_name == "hr-bot" and t.model == "gpt-4.1" and t.service_version == "1.4.0"
    tree = {s["span_id"]: s for s in r.raw["spans"]}
    assert tree["00f067aa0ba902b7"]["parent_id"] == "b7ad6b7169203331"   # relationships kept
    assert not t.warnings


def test_openinference_trace_with_retrieval():
    trace = _otlp_json([
        {"trace": T, "id": "a1", "name": "AgentRunner", "start": 0, "end": 2_000_000_000,
         "attrs": {"openinference.span.kind": "AGENT", "input.value": json.dumps({"input": "VPN rules?"}),
                   "output.value": "Use the corporate VPN [it_security.md]."}},
        {"trace": T, "id": "a2", "parent": "a1", "name": "retrieve", "start": 1, "end": 2,
         "attrs": {"openinference.span.kind": "RETRIEVER", "input.value": "VPN rules",
                   "retrieval.documents.0.document.id": "it_security.md#2",
                   "retrieval.documents.0.document.content": "Always use the VPN.",
                   "retrieval.documents.0.document.score": 0.91,
                   "retrieval.documents.1.document.id": "remote_work.md#0",
                   "retrieval.documents.1.document.content": "Remote work rules."}},
        {"trace": T, "id": "a3", "parent": "a1", "name": "llm", "start": 3, "end": 4,
         "attrs": {"openinference.span.kind": "LLM", "llm.model_name": "llama-3",
                   "llm.token_count.prompt": 50, "llm.token_count.completion": 10}},
    ])
    [t] = normalize_spans(parse_otlp(trace))
    assert t.question == "VPN rules?" and "corporate VPN" in t.result.answer
    assert [d.id for d in t.result.retrieved] == ["it_security.md#2", "remote_work.md#0"]
    assert t.result.retrieved[0].score == 0.91
    assert t.result.prompt_tokens == 50 and t.model == "llama-3"


def test_openllmetry_prompt_completion_attributes_and_errors():
    trace = _otlp_json([
        {"trace": T, "id": "c1", "name": "openai.chat", "start": 0, "end": 10, "status": 2,
         "attrs": {"gen_ai.prompt.0.role": "system", "gen_ai.prompt.0.content": "be nice",
                   "gen_ai.prompt.1.role": "user", "gen_ai.prompt.1.content": "hello?",
                   "llm.usage.prompt_tokens": 5, "llm.usage.completion_tokens": 0}}])
    [t] = normalize_spans(parse_otlp(trace))
    assert t.question == "hello?"
    assert t.result.error and "openai.chat" in t.result.error      # failed span with no answer
    assert "no output found" in " ".join(t.warnings)


def test_partial_trace_warns_but_still_normalizes():
    trace = _otlp_json([{"trace": T, "id": "d2", "parent": "missing-parent", "name": "chat",
                         "start": 0, "end": 1, "attrs": {"gen_ai.operation.name": "chat",
                                                          "gen_ai.output.messages": json.dumps([{"role": "assistant", "content": "hi"}])}}])
    [t] = normalize_spans(parse_otlp(trace))
    assert t.result.answer == "hi" and any("partial trace" in w for w in t.warnings)


def test_invalid_payload_rejected():
    with pytest.raises(ValueError):
        parse_otlp({"hello": "world"})


def test_replay_agent_end_to_end(tmp_path):
    (tmp_path / "t1.json").write_text(json.dumps(genai_trace()), encoding="utf-8")
    other = genai_trace(trace="1" * 32, question="Is there a gym?", answer="I don't know.")
    (tmp_path / "t2.jsonl").write_text(json.dumps(other) + "\n", encoding="utf-8")
    agent = REGISTRY.build("system", {"type": "otel_replay", "traces": [str(tmp_path / "*.json*")]})
    r = agent.run("  how many DAYS of leave? ")
    assert "24 days" in r.answer and r.trace[0].tool == "search_policies"
    assert agent.run("Is there a gym?").answer == "I don't know."
    assert agent.run("unknown question").error == "no recorded trace for this input"
    assert agent.health_check().ok


def test_evalkit_run_on_recorded_traces(tmp_path):
    """Full pipeline: traces -> otel_replay system -> scored report, no adapter code."""
    import yaml
    from evalkit import runner
    (tmp_path / "traces.json").write_text(json.dumps(genai_trace()), encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(json.dumps(
        {"id": "q1", "input": "How many days of leave?", "expected_contains": ["24 days"],
         "expected_tools": ["search_policies"]}) + "\n", encoding="utf-8")
    cfg = {"name": "otel", "system": {"type": "otel_replay", "traces": str(tmp_path / "traces.json")},
           "dataset": {"path": str(tmp_path / "cases.jsonl")}, "output_dir": str(tmp_path / "out"),
           "run": {"concurrency": 1}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    report, _ = runner.run(str(tmp_path / "c.yaml"), log=lambda *a: None)
    o = report["overall"]
    assert o["answer_contains"]["mean"] == 1.0 and o["tool_selection"]["mean"] == 1.0
    assert o["total_tokens"]["mean"] == 375.0


# ---------- against the real OpenTelemetry SDK ----------
otel_sdk = pytest.importorskip("opentelemetry.sdk.trace", reason="opentelemetry-sdk not installed")


def _real_spans():
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import Status, StatusCode
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "real-agent", "service.version": "2.0"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("evalkit-test")
    with tracer.start_as_current_span("invoke_agent support") as root:
        root.set_attribute("gen_ai.operation.name", "invoke_agent")
        root.set_attribute("gen_ai.agent.name", "support")
        root.set_attribute("gen_ai.input.messages", json.dumps([{"role": "user", "parts": [{"type": "text", "content": "Reset my password?"}]}]))
        with tracer.start_as_current_span("chat gpt-4o") as llm:
            llm.set_attribute("gen_ai.operation.name", "chat")
            llm.set_attribute("gen_ai.request.model", "gpt-4o")
            llm.set_attribute("gen_ai.usage.input_tokens", 42)
            llm.set_attribute("gen_ai.usage.output_tokens", 7)
        with tracer.start_as_current_span("execute_tool reset") as tool:
            tool.set_attribute("gen_ai.operation.name", "execute_tool")
            tool.set_attribute("gen_ai.tool.name", "reset_password")
            tool.set_attribute("gen_ai.tool.call.arguments", json.dumps({"user": "u1"}))
            tool.set_status(Status(StatusCode.OK))
        root.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": "Done: check your email."}]}]))
    return exporter.get_finished_spans()


def test_real_sdk_otlp_protobuf():
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    payload = encode_spans(_real_spans()).SerializeToString()       # what OTLP/HTTP protobuf sends
    [t] = normalize_spans(parse_otlp(payload))
    assert t.question == "Reset my password?" and t.result.answer == "Done: check your email."
    assert [c.tool for c in t.result.trace] == ["reset_password"] and t.result.trace[0].args == {"user": "u1"}
    assert t.result.prompt_tokens == 42 and t.result.completion_tokens == 7
    assert t.service == "real-agent" and t.service_version == "2.0" and len(t.trace_id) == 32
    assert len(t.result.raw["spans"]) == 3


def test_real_sdk_otlp_json():
    from google.protobuf.json_format import MessageToJson
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    text = MessageToJson(encode_spans(_real_spans()))                # OTLP/JSON (ids base64 here)
    [t] = normalize_spans(parse_otlp(text))
    assert t.result.answer == "Done: check your email." and t.result.prompt_tokens == 42


def test_load_traces_protobuf_file(tmp_path):
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    (tmp_path / "batch.pb").write_bytes(encode_spans(_real_spans()).SerializeToString())
    assert len(load_traces(str(tmp_path / "*.pb"))) == 3
