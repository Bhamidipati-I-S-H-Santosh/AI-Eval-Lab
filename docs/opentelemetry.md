# Evaluating agents from OpenTelemetry traces

Any agent that emits OpenTelemetry traces can be evaluated without an
adapter: instrument it once (most frameworks already can), export the traces,
and evalkit turns each trace into the same result an adapter would produce.

## Supported conventions

| Convention | Used by | What evalkit reads |
|---|---|---|
| OpenTelemetry GenAI semantic conventions | OpenAI/Azure/Google SDK instrumentations, Semantic Kernel, Microsoft Agent Framework, AutoGen, Strands, many others | `gen_ai.operation.name` (`invoke_agent`, `chat`, `execute_tool`...), `gen_ai.usage.input_tokens/output_tokens` (and older `prompt_tokens/completion_tokens`), models, agent name, tool name/arguments/result, `gen_ai.input.messages` / `gen_ai.output.messages`, `gen_ai.user.message` / `gen_ai.choice` events |
| OpenInference | Arize Phoenix, LlamaIndex, LangChain, DSPy, Haystack instrumentations | `openinference.span.kind`, `input.value` / `output.value`, `llm.token_count.*`, `tool.name` / `tool.parameters`, `retrieval.documents.N.document.*` |
| OpenLLMetry (Traceloop) | Traceloop SDK | `gen_ai.prompt.N.*` / `gen_ai.completion.N.*`, `llm.usage.*`, `traceloop.entity.input/output` |

Per trace evalkit extracts: the question (agent span input, else the first
user message an LLM saw), the answer, tool calls with arguments and results,
retrieved documents with ids and scores, total tokens across LLM spans,
latency (first span start to last span end), errors (span status), model,
agent name, `service.name` / `service.version`, and the span tree with
parent/child links. Partial traces and missing fields produce warnings, not
failures.

Formats: OTLP/JSON (`resourceSpans`), OTLP/protobuf (needs `evalkit[otel]`),
JSON lists or JSON-lines of spans. Ids may be hex or base64.

## Three ways to use it

**1. Evaluate recorded traces against a dataset** (no adapter):

```yaml
system: {type: otel_replay, traces: ["traces/*.json"]}
dataset: {path: golden.jsonl}       # questions are matched to traces by their input
```

**2. Build a dataset from production traces**, then add expected answers:

```bash
evalkit otel import traces/*.json --out datasets/from-prod.jsonl
```

**3. Send traces to the evalkit server** and monitor them continuously:

```bash
# Point any OTLP/HTTP exporter at the server:
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://evalkit.corp.example/v1/traces
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=<ingest key>"
```

or run an OpenTelemetry Collector in front of it (filters to GenAI spans and
batches): see `deploy/otel-collector.yaml`. Then `evalkit monitor CONFIG`
([monitoring.md](monitoring.md)).

The receiver requires a principal with the `ingest` role, validates every
payload, stores it in the configured storage under `otel/YYYY/MM/DD/`, and
answers in the encoding it received (OTLP/HTTP spec).

## Content capture

Many instrumentations do not record prompt/response text by default, because
it may contain sensitive data. Answers and questions are needed for quality
metrics, so enable content capture in the instrumentation (for example
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`) only where that is
approved, and consider `privacy.redact_stored` on the evalkit side.
