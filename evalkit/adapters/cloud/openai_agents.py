"""OpenAI hosted agents: the Responses API and the Assistants API.

config (Responses API, POST /v1/responses):
  type: openai_responses
  model: gpt-4.1
  instructions: "You are the Northwind support agent."   # optional system prompt
  tools:                                    # optional, passed to the API as-is
    - { type: file_search, vector_store_ids: [vs_abc123] }
    - { type: web_search_preview }
  api_key: ${OPENAI_API_KEY}                # default: the OPENAI_API_KEY env var
  base_url: null                            # optional OpenAI-compatible gateway
  store: true                               # needed for multi-turn via previous_response_id
  timeout: 120

config (Assistants API v2, /v1/threads + /v1/threads/{id}/runs):
  type: openai_assistants
  assistant_id: asst_abc123
  api_key: ${OPENAI_API_KEY}
  timeout: 120                              # max seconds to wait for one run
  poll_interval: 1.0
  delete_threads: true                      # delete each thread when the test case ends

Function tools: both APIs hand function calls back to the caller. From Python
pass tool_handlers={"tool_name": fn(args: dict) -> str}; without a handler the
call is recorded in the trace and the result carries an error.
Needs: pip install 'evalkit[openai]'.
"""
from __future__ import annotations

import importlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ...errors import ConfigError, PluginDependencyError
from ...plugins import HealthStatus, timed_health
from ...types import AgentResult, RetrievedDoc, ToolCall
from ..base import AgentAdapter

log = logging.getLogger("evalkit.adapters.cloud")

_MAX_RESULT_CHARS = 4000


# ---------- helpers shared by the SDK-object based adapters (also used by azure_foundry) ----------

def _require(module: str, plugin: str, extra: str):
    """Import a vendor SDK lazily, turning a missing package into an actionable error."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise PluginDependencyError(plugin, e.name or module, extra) from e


def _g(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK model or a plain dict; None counts as missing."""
    if obj is None:
        return default
    v = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return default if v is None else v


def _enum(value: Any) -> str:
    """Status/role values may be plain strings or SDK enums (str subclasses with .value)."""
    return str(getattr(value, "value", value) or "").lower()


def _items(page: Any) -> list:
    """SDK list results come as pages (with .data) or iterables."""
    if page is None:
        return []
    data = _g(page, "data")
    if data is not None:
        return list(data)
    return list(page)


def _plain(obj: Any, depth: int = 0) -> Any:
    """Best-effort JSON-safe copy of an SDK object."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if depth > 8:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _plain(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v, depth + 1) for v in obj]
    for method in ("model_dump", "as_dict", "to_dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return _plain(fn(), depth + 1)
            except Exception:  # noqa: BLE001 - fall through to the next strategy
                pass
    if hasattr(obj, "__dict__"):
        return _plain({k: v for k, v in vars(obj).items() if not k.startswith("_")}, depth + 1)
    return str(obj)


def _json_args(value: Any) -> dict:
    """Tool-call arguments arrive as a JSON string or a dict."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return _plain(value)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {"raw": str(value)}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _text(value: Any) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else json.dumps(_plain(value), ensure_ascii=False, default=str)
    return s[:_MAX_RESULT_CHARS]


def _add_doc(docs: list[RetrievedDoc], doc: RetrievedDoc) -> None:
    """Append, merging with an existing doc of the same id (a cited doc that
    was also returned by a search keeps the search text and score)."""
    for d in docs:
        if d.id == doc.id:
            if not d.text and doc.text:
                d.text = doc.text
            if d.score is None:
                d.score = doc.score
            for k, v in doc.metadata.items():
                if v is not None and d.metadata.get(k) in (None, False, ""):
                    d.metadata[k] = v
            return
    docs.append(doc)


def _content_text(content: Any) -> str:
    """file_search result content: a list of {type: text, text: ...} parts."""
    if isinstance(content, str):
        return content
    return "\n".join(str(_g(p, "text", "")) for p in (content or []) if _g(p, "text"))


def _annotation_doc(ann: Any) -> RetrievedDoc | None:
    """Map a message annotation (citation) to a RetrievedDoc."""
    typ = _enum(_g(ann, "type", ""))
    if typ == "url_citation":
        uc = _g(ann, "url_citation", ann)       # Assistants/Azure nest it; Responses is flat
        url = _g(uc, "url", "")
        if not url:
            return None
        return RetrievedDoc(id=str(url), text=str(_g(uc, "title", "")),
                            metadata={"type": "url_citation", "url": url, "title": _g(uc, "title"),
                                      "cited": True})
    if typ in ("file_citation", "container_file_citation"):
        fc = _g(ann, "file_citation", ann)
        fid = _g(fc, "file_id") or _g(ann, "file_id")
        if not fid:
            return None
        return RetrievedDoc(id=str(fid), text=str(_g(fc, "quote", "") or ""),
                            metadata={"type": typ, "filename": _g(fc, "filename") or _g(ann, "filename"),
                                      "cited": True})
    return None


def _query_from_url(url: str) -> str | None:
    from urllib.parse import parse_qs, urlparse
    try:
        return (parse_qs(urlparse(url).query).get("q") or [None])[0]
    except ValueError:
        return None


def _step_tool_call(tc: Any) -> tuple[ToolCall, list[RetrievedDoc]]:
    """One run-step tool call (Assistants API / Azure Agent Service) -> ToolCall + docs."""
    typ = _enum(_g(tc, "type", "tool")) or "tool"
    if typ == "function":
        f = _g(tc, "function", {})
        return ToolCall(str(_g(f, "name", "function")), _json_args(_g(f, "arguments")),
                        _text(_g(f, "output"))), []
    if typ == "file_search":
        fs = _g(tc, "file_search", {})
        results = _g(fs, "results", []) or []
        docs = [RetrievedDoc(id=str(_g(r, "file_id") or _g(r, "file_name") or f"result-{i}"),
                             text=_content_text(_g(r, "content")), score=_g(r, "score"),
                             metadata={"file_name": _g(r, "file_name"), "source": "file_search"})
                for i, r in enumerate(results)]
        return ToolCall("file_search", {}, f"{len(docs)} result(s)"), docs
    if typ == "code_interpreter":
        ci = _g(tc, "code_interpreter", {})
        logs = [str(_g(o, "logs")) for o in (_g(ci, "outputs", []) or []) if _g(o, "logs")]
        return ToolCall("code_interpreter", {"input": _g(ci, "input", "")}, "\n".join(logs)[:_MAX_RESULT_CHARS]), []
    # Hosted tools (azure_ai_search, bing_grounding, sharepoint_grounding, openapi,
    # connected_agent, mcp, ...): details sit under a key named after the type.
    full = _plain(tc)
    full = full if isinstance(full, dict) else {}
    detail = full.get(typ)
    merged = {k: v for k, v in full.items() if k not in ("id", "type", typ)}
    if isinstance(detail, dict):
        merged.update(detail)
    name = typ
    if typ in ("mcp", "connected_agent", "azure_function", "openapi") and merged.get("name"):
        name = str(merged["name"])
    if "arguments" in merged:
        args = _json_args(merged["arguments"])
    elif merged.get("requesturl") or merged.get("request_url"):
        url = merged.get("requesturl") or merged.get("request_url")
        args = {"request_url": url, "query": _query_from_url(str(url))}
    elif "input" in merged:
        args = {"query": merged["input"]}
    else:
        args = {k: v for k, v in merged.items() if k not in ("output", "response", "name")}
    result = merged.get("output", merged.get("response", ""))
    return ToolCall(name, args, _text(result)), []


def _message_text(msg: Any, strip_markers: bool) -> tuple[str, list[RetrievedDoc]]:
    """Assistant message (Assistants API / Azure) -> text + cited docs."""
    texts, docs = [], []
    for part in _g(msg, "content", []) or []:
        if _enum(_g(part, "type")) != "text":
            continue
        t = _g(part, "text", {})
        value = str(_g(t, "value", "") if not isinstance(t, str) else t)
        for ann in _g(t, "annotations", []) or []:
            d = _annotation_doc(ann)
            if d:
                _add_doc(docs, d)
            marker = _g(ann, "text")
            if strip_markers and marker and isinstance(marker, str):
                value = value.replace(marker, "")
        texts.append(value)
    out = "\n".join(texts)
    if strip_markers:
        out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
    return out.strip(), docs


@dataclass
class _Acc:
    """Accumulates one or more turns into an AgentResult."""
    answer: str = ""
    trace: list[ToolCall] = field(default_factory=list)
    retrieved: list[RetrievedDoc] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    have_tokens: bool = False
    error: str | None = None
    raw: dict = field(default_factory=dict)

    def tokens(self, prompt: Any, completion: Any) -> None:
        if prompt is None and completion is None:
            return
        self.have_tokens = True
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)

    def docs(self, docs: list[RetrievedDoc]) -> None:
        for d in docs:
            _add_doc(self.retrieved, d)

    def result(self, t0: float) -> AgentResult:
        return AgentResult(answer=self.answer, trace=self.trace, retrieved=self.retrieved,
                           prompt_tokens=self.prompt_tokens if self.have_tokens else None,
                           completion_tokens=self.completion_tokens if self.have_tokens else None,
                           latency_s=round(time.monotonic() - t0, 3), error=self.error, raw=self.raw)


def _call_handler(handlers: dict[str, Callable], name: str, args: dict) -> str:
    out = handlers[name](args)
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)


# ---------- Responses API ----------

class OpenAIResponsesAgent(AgentAdapter):
    """OpenAI Responses API (client.responses.create). Multi-turn conversations
    chain turns with previous_response_id (requires store: true)."""

    def __init__(self, model: str | None = None, instructions: str | None = None,
                 tools: list[dict] | None = None, include: list[str] | None = None,
                 prompt: dict | None = None, temperature: float | None = None,
                 max_output_tokens: int | None = None, reasoning: dict | None = None,
                 store: bool = True, extra_params: dict | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 organization: str | None = None, project: str | None = None,
                 timeout: float = 120, max_tool_rounds: int = 8,
                 tool_handlers: dict[str, Callable] | None = None, client=None, **_):
        if not model and not prompt:
            raise ConfigError("openai_responses needs 'model' (e.g. gpt-4.1) or a stored "
                              "'prompt: {id: pmpt_...}'")
        if client is None:
            openai = _require("openai", "system:openai_responses", "openai")
            client = openai.OpenAI(api_key=api_key or None, base_url=base_url or None,
                                   organization=organization or None, project=project or None,
                                   timeout=timeout)
        self.client = client
        self.model, self.instructions, self.tools, self.prompt = model, instructions, tools, prompt
        if include is None and any((t or {}).get("type") == "file_search" for t in (tools or [])):
            include = ["file_search_call.results"]     # return the retrieved chunks
        self.include = include
        self.temperature, self.max_output_tokens, self.reasoning = temperature, max_output_tokens, reasoning
        self.store, self.extra_params = store, dict(extra_params or {})
        self.max_tool_rounds = max_tool_rounds
        self.tool_handlers = dict(tool_handlers or {})

    def _create(self, input: Any, previous_response_id: str | None):
        kw: dict[str, Any] = {"input": input, "store": self.store}
        for k in ("model", "instructions", "tools", "include", "prompt", "temperature",
                  "max_output_tokens", "reasoning"):
            v = getattr(self, k)
            if v is not None:
                kw[k] = v
        if previous_response_id:
            kw["previous_response_id"] = previous_response_id
        kw.update(self.extra_params)
        return self.client.responses.create(**kw)

    def _parse(self, resp: Any, acc: _Acc) -> tuple[str, list]:
        """Parse response.output items. Returns (answer text, pending function calls)."""
        texts, pending = [], []
        for item in _g(resp, "output", []) or []:
            typ = _g(item, "type", "")
            if typ == "message":
                for part in _g(item, "content", []) or []:
                    ptype = _g(part, "type")
                    if ptype in ("output_text", "text"):
                        texts.append(str(_g(part, "text", "")))
                        for ann in _g(part, "annotations", []) or []:
                            d = _annotation_doc(ann)
                            if d:
                                _add_doc(acc.retrieved, d)
                    elif ptype == "refusal":
                        texts.append(str(_g(part, "refusal", "")))
            elif typ == "function_call":
                args = _json_args(_g(item, "arguments"))
                acc.trace.append(ToolCall(str(_g(item, "name", "function")), args))
                pending.append(item)
            elif typ == "file_search_call":
                results = _g(item, "results", []) or []
                for i, r in enumerate(results):
                    _add_doc(acc.retrieved, RetrievedDoc(
                        id=str(_g(r, "file_id") or _g(r, "filename") or f"result-{i}"),
                        text=str(_g(r, "text", "")), score=_g(r, "score"),
                        metadata={"filename": _g(r, "filename"), "attributes": _plain(_g(r, "attributes")),
                                  "source": "file_search"}))
                acc.trace.append(ToolCall("file_search", {"queries": list(_g(item, "queries", []) or [])},
                                          f"{len(results)} result(s)"))
            elif typ == "web_search_call":
                action = _g(item, "action", {})
                sources = _g(action, "sources", []) or []
                for s in sources:
                    if _g(s, "url"):
                        _add_doc(acc.retrieved, RetrievedDoc(id=str(_g(s, "url")), text="",
                                                             metadata={"source": "web_search"}))
                acc.trace.append(ToolCall("web_search", {"query": _g(action, "query")},
                                          f"{len(sources)} source(s)" if sources else str(_g(item, "status", ""))))
            elif typ == "code_interpreter_call":
                outs = [str(_g(o, "logs")) for o in (_g(item, "outputs", []) or []) if _g(o, "logs")]
                acc.trace.append(ToolCall("code_interpreter", {"code": _g(item, "code", "")},
                                          "\n".join(outs)[:_MAX_RESULT_CHARS]))
            elif typ == "mcp_call":
                args = _json_args(_g(item, "arguments"))
                args.setdefault("server_label", _g(item, "server_label"))
                acc.trace.append(ToolCall(str(_g(item, "name", "mcp")), args,
                                          _text(_g(item, "output") or _g(item, "error"))))
            elif isinstance(typ, str) and typ.endswith("_call"):
                acc.trace.append(ToolCall(typ[:-5], {}, str(_g(item, "status", ""))))
        answer = _g(resp, "output_text")
        if not isinstance(answer, str) or not answer:
            answer = "\n".join(t for t in texts if t)
        return answer.strip(), pending

    def _turn(self, input: Any, previous_response_id: str | None, acc: _Acc) -> str | None:
        """Run one user turn (including function-call rounds). Returns the last response id."""
        ids = acc.raw.setdefault("response_ids", [])
        for _round in range(self.max_tool_rounds + 1):
            resp = self._create(input, previous_response_id)
            rid = _g(resp, "id")
            ids.append(rid)
            usage = _g(resp, "usage")
            acc.tokens(_g(usage, "input_tokens"), _g(usage, "output_tokens"))
            answer, pending = self._parse(resp, acc)
            acc.answer = answer
            status = _g(resp, "status", "completed")
            acc.raw["status"] = status
            if status in ("failed", "cancelled"):
                err = _g(resp, "error", {})
                acc.error = f"OpenAI response {rid} {status}: {_g(err, 'code', '')} {_g(err, 'message', '')}".strip()
                return rid
            if status == "incomplete":
                acc.raw["incomplete_reason"] = _g(_g(resp, "incomplete_details", {}), "reason")
            if not pending:
                return rid
            missing = [_g(p, "name") for p in pending if _g(p, "name") not in self.tool_handlers]
            if missing:
                acc.error = (f"model called function tool(s) {missing} but no tool_handlers were given "
                             f"for them; pass tool_handlers={{name: fn}} from Python")
                return rid
            outputs = []
            for p in pending:
                name, args = _g(p, "name"), _json_args(_g(p, "arguments"))
                out = _call_handler(self.tool_handlers, name, args)
                for t in reversed(acc.trace):           # attach the result to the recorded call
                    if t.tool == name and not t.result:
                        t.result = out[:_MAX_RESULT_CHARS]
                        break
                outputs.append({"type": "function_call_output", "call_id": _g(p, "call_id"), "output": out})
            if self.store:
                input, previous_response_id = outputs, rid
            else:  # no server-side state: resend the whole exchange
                base = input if isinstance(input, list) else [{"role": "user", "content": input}]
                input = base + [_plain(i) for i in (_g(resp, "output", []) or [])] + outputs
                previous_response_id = None
        acc.error = f"stopped after {self.max_tool_rounds} function-call rounds without a final answer"
        return previous_response_id

    def run(self, input: str) -> AgentResult:
        t0, acc = time.monotonic(), _Acc()
        self._turn(input, None, acc)
        acc.raw["model"] = self.model
        return acc.result(t0)

    def run_conversation(self, turns: list[str]) -> AgentResult:
        t0, acc = time.monotonic(), _Acc()
        prev: str | None = None
        history: list[dict] = []
        for turn in turns:
            if self.store:
                prev = self._turn(turn, prev, acc)
            else:
                history.append({"role": "user", "content": turn})
                self._turn(list(history), None, acc)
                history.append({"role": "assistant", "content": acc.answer})
            if acc.error:
                break
        acc.raw.update(model=self.model, turns=len(turns))
        return acc.result(t0)

    def health_check(self) -> HealthStatus:
        if not self.model:
            return super().health_check()
        return timed_health(lambda: f"model {_g(self.client.models.retrieve(self.model), 'id', self.model)} reachable")

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


# ---------- thread/run engine (Assistants API; reused by Azure AI Foundry Agent Service) ----------

_TERMINAL = {"completed", "failed", "cancelled", "expired", "incomplete"}


class _ThreadRunAgent(AgentAdapter):
    """Shared thread -> message -> run -> poll -> messages + run steps flow.
    Subclasses implement the _op_* methods for their SDK."""

    service = "agent"
    timeout: float = 120
    poll_interval: float = 1.0
    delete_threads: bool = True
    strip_citation_markers: bool = True
    tool_handlers: dict[str, Callable]
    auto_approve_tools: bool = False

    # --- SDK operations (subclass) ---
    def _op_create_thread(self) -> Any: raise NotImplementedError
    def _op_post_message(self, thread_id: str, text: str) -> Any: raise NotImplementedError
    def _op_create_run(self, thread_id: str) -> Any: raise NotImplementedError
    def _op_get_run(self, thread_id: str, run_id: str) -> Any: raise NotImplementedError
    def _op_cancel_run(self, thread_id: str, run_id: str) -> Any: raise NotImplementedError
    def _op_submit(self, thread_id: str, run_id: str, **kw) -> Any: raise NotImplementedError
    def _op_list_messages(self, thread_id: str, run_id: str) -> list: raise NotImplementedError
    def _op_list_steps(self, thread_id: str, run_id: str) -> list: raise NotImplementedError
    def _op_delete_thread(self, thread_id: str) -> Any: raise NotImplementedError

    def _process_run(self, thread_id: str) -> tuple[Any, str | None]:
        """Create a run and wait for a terminal state -> (run, evalkit error or None)."""
        return self._wait(thread_id, self._op_create_run(thread_id))

    def _cancel_quietly(self, thread_id: str, run_id: str) -> None:
        try:
            self._op_cancel_run(thread_id, run_id)
        except Exception as e:  # noqa: BLE001 - the primary error is reported by the caller
            log.warning("%s: could not cancel run %s: %s", self.service, run_id, e)

    def _handle_required_action(self, thread_id: str, run: Any) -> tuple[Any, str | None]:
        ra = _g(run, "required_action", {})
        rtype = _g(ra, "type", "submit_tool_outputs")
        run_id = _g(run, "id")
        if rtype == "submit_tool_approval":
            calls = _g(_g(ra, "submit_tool_approval", {}), "tool_calls", []) or []
            if not self.auto_approve_tools:
                self._cancel_quietly(thread_id, run_id)
                names = [_g(c, "name") or _g(c, "type") for c in calls]
                return run, (f"run {run_id} is waiting for tool approval {names}; set "
                             f"auto_approve_tools: true to approve automatically during evaluation")
            approvals = [{"tool_call_id": _g(c, "id"), "approve": True} for c in calls]
            return self._op_submit(thread_id, run_id, tool_approvals=approvals), None
        calls = _g(_g(ra, "submit_tool_outputs", {}), "tool_calls", []) or []
        names = [_g(_g(c, "function", {}), "name") for c in calls]
        missing = [n for n in names if n not in self.tool_handlers]
        if missing:
            self._cancel_quietly(thread_id, run_id)
            return run, (f"run {run_id} requires function tool(s) {missing} that run client-side; "
                         f"pass tool_handlers={{name: fn}} from Python (run cancelled)")
        outputs = []
        for c in calls:
            f = _g(c, "function", {})
            out = _call_handler(self.tool_handlers, _g(f, "name"), _json_args(_g(f, "arguments")))
            outputs.append({"tool_call_id": _g(c, "id"), "output": out})
        return self._op_submit(thread_id, run_id, tool_outputs=outputs), None

    def _wait(self, thread_id: str, run: Any) -> tuple[Any, str | None]:
        """Poll until the run is terminal; bounded by self.timeout."""
        deadline = time.monotonic() + self.timeout
        while True:
            status = _enum(_g(run, "status", ""))
            if status in _TERMINAL:
                return run, None
            if status == "requires_action":
                run, err = self._handle_required_action(thread_id, run)
                if err:
                    return run, err
                continue
            if time.monotonic() >= deadline:
                self._cancel_quietly(thread_id, _g(run, "id"))
                raise TimeoutError(f"{self.service} run {_g(run, 'id')} on thread {thread_id} did not finish "
                                   f"within {self.timeout}s (last status '{status}'); run cancelled. "
                                   f"Raise 'timeout' if the agent is legitimately slow.")
            if self.poll_interval:
                time.sleep(self.poll_interval)
            run = self._op_get_run(thread_id, _g(run, "id"))

    def _turn(self, thread_id: str, text: str, acc: _Acc) -> None:
        self._op_post_message(thread_id, text)
        run, err = self._process_run(thread_id)
        run_id = _g(run, "id")
        acc.raw.setdefault("run_ids", []).append(run_id)
        status = _enum(_g(run, "status", ""))
        usage = _g(run, "usage")
        acc.tokens(_g(usage, "prompt_tokens"), _g(usage, "completion_tokens"))
        # tool calls and retrieved chunks from the run steps
        for step in self._op_list_steps(thread_id, run_id):
            det = _g(step, "step_details", {})
            if _enum(_g(det, "type", "")) != "tool_calls":
                continue
            for tc in _g(det, "tool_calls", []) or []:
                call, docs = _step_tool_call(tc)
                acc.trace.append(call)
                acc.docs(docs)
        # the assistant's reply for this run
        texts = []
        for msg in self._op_list_messages(thread_id, run_id):
            if _enum(_g(msg, "role", "")) != "assistant":
                continue
            if _g(msg, "run_id") not in (None, run_id):
                continue
            t, docs = _message_text(msg, self.strip_citation_markers)
            texts.append(t)
            acc.docs(docs)
        acc.answer = "\n".join(t for t in texts if t)
        acc.raw["status"] = status
        if err:
            acc.error = err
        elif status != "completed":
            le = _g(run, "last_error") or _g(run, "incomplete_details")
            detail = f"{_g(le, 'code', '')} {_g(le, 'message', '') or _g(le, 'reason', '')}".strip()
            acc.error = f"{self.service} run {run_id} ended with status '{status}'" + (f": {detail}" if detail else "")

    def _finish_thread(self, thread_id: str) -> None:
        if not self.delete_threads:
            return
        try:
            self._op_delete_thread(thread_id)
        except Exception as e:  # noqa: BLE001 - cleanup must not hide the result, but is logged
            log.warning("%s: could not delete thread %s: %s", self.service, thread_id, e)

    def run(self, input: str) -> AgentResult:
        return self.run_conversation([input])

    def run_conversation(self, turns: list[str]) -> AgentResult:
        t0, acc = time.monotonic(), _Acc()
        thread_id = _g(self._op_create_thread(), "id")
        acc.raw["thread_id"] = thread_id
        try:
            for turn in turns:
                self._turn(thread_id, turn, acc)
                if acc.error:
                    break
        finally:
            self._finish_thread(thread_id)
        if len(turns) > 1:
            acc.raw["turns"] = len(turns)
        return acc.result(t0)


class OpenAIAssistantsAgent(_ThreadRunAgent):
    """OpenAI Assistants API v2: one thread per test case, a run per user turn."""

    service = "OpenAI Assistants"

    def __init__(self, assistant_id: str, additional_instructions: str | None = None,
                 model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 organization: str | None = None, project: str | None = None,
                 timeout: float = 120, poll_interval: float = 1.0, delete_threads: bool = True,
                 include_search_content: bool = True, strip_citation_markers: bool = True,
                 tool_handlers: dict[str, Callable] | None = None, client=None, **_):
        if not assistant_id:
            raise ConfigError("openai_assistants needs 'assistant_id' (asst_...)")
        if client is None:
            openai = _require("openai", "system:openai_assistants", "openai")
            client = openai.OpenAI(api_key=api_key or None, base_url=base_url or None,
                                   organization=organization or None, project=project or None)
        self.client, self.assistant_id = client, assistant_id
        self.additional_instructions, self.model = additional_instructions, model
        self.timeout, self.poll_interval, self.delete_threads = float(timeout), float(poll_interval), delete_threads
        self.include_search_content, self.strip_citation_markers = include_search_content, strip_citation_markers
        self.tool_handlers = dict(tool_handlers or {})

    @property
    def _threads(self):
        return self.client.beta.threads

    def _op_create_thread(self):
        return self._threads.create()

    def _op_post_message(self, thread_id, text):
        return self._threads.messages.create(thread_id=thread_id, role="user", content=text)

    def _op_create_run(self, thread_id):
        kw: dict[str, Any] = {"thread_id": thread_id, "assistant_id": self.assistant_id}
        if self.additional_instructions:
            kw["additional_instructions"] = self.additional_instructions
        if self.model:
            kw["model"] = self.model
        return self._threads.runs.create(**kw)

    def _op_get_run(self, thread_id, run_id):
        return self._threads.runs.retrieve(run_id=run_id, thread_id=thread_id)

    def _op_cancel_run(self, thread_id, run_id):
        return self._threads.runs.cancel(run_id=run_id, thread_id=thread_id)

    def _op_submit(self, thread_id, run_id, **kw):
        return self._threads.runs.submit_tool_outputs(run_id=run_id, thread_id=thread_id, **kw)

    def _op_list_messages(self, thread_id, run_id):
        return _items(self._threads.messages.list(thread_id=thread_id, run_id=run_id, order="asc"))

    def _op_list_steps(self, thread_id, run_id):
        kw: dict[str, Any] = {"run_id": run_id, "thread_id": thread_id, "order": "asc"}
        if self.include_search_content:
            kw["include"] = ["step_details.tool_calls[*].file_search.results[*].content"]
        return _items(self._threads.runs.steps.list(**kw))

    def _op_delete_thread(self, thread_id):
        return self._threads.delete(thread_id=thread_id)

    def health_check(self) -> HealthStatus:
        def probe():
            a = self.client.beta.assistants.retrieve(assistant_id=self.assistant_id)
            return f"assistant '{_g(a, 'name', self.assistant_id)}' ({_g(a, 'model', '?')}) reachable"
        return timed_health(probe)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
