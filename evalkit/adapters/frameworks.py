"""Evaluate agents built with popular frameworks, in-process, without changing them.

Each adapter imports its framework lazily and only duck-types the objects it
is given, so evalkit itself never depends on the framework.

  system:
    type: langchain                      # LangChain / LangGraph Runnable
    target: my_app.agent:executor        # object, or zero-arg factory returning one
    input_key: input                     # or  input_format: messages  for LangGraph message graphs
    output_key: output                   # optional dotted path into a dict output
    retriever_hook: my_app.agent:set_search   # optional, enables injection tests

  system: { type: llamaindex, target: my_app.rag:query_engine, token_counter: my_app.rag:token_counter }
  system: { type: crewai, target: my_app.crew:build_crew, input_key: topic }
  system: { type: semantic_kernel, target: my_app.sk:kernel, function: policies.answer, input_arg: input }
  system: { type: semantic_kernel, target: my_app.sk:agent }   # ChatCompletionAgent

See docs/integrations/frameworks.md for what each adapter captures.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Callable

from ..errors import ConfigError
from ..plugins import HealthStatus, timed_health
from ..types import AgentResult, RetrievedDoc, ToolCall
from ..utils import get_path, import_object
from .base import AgentAdapter

_TEXT_LIMIT = 20_000  # chars of tool results / retrieved text kept per item


# ---------- shared helpers ----------
def _resolve_target(target: Any, methods: tuple[str, ...], label: str) -> Any:
    """'module:attr' -> object. A zero-arg factory is called once. The result
    must have at least one of `methods`."""
    obj = import_object(target) if isinstance(target, str) else target
    if not any(hasattr(obj, m) for m in methods) and callable(obj):
        obj = obj()
    if not any(hasattr(obj, m) for m in methods):
        raise ConfigError(f"{label}: target {target!r} resolved to {type(obj).__name__}, which has none of "
                          f"{list(methods)}. Point 'target' at the framework object (or a zero-arg function "
                          f"returning it), as 'module:attribute'")
    return obj


def _optional_object(path: Any) -> Any:
    return import_object(path) if isinstance(path, str) else path


def _text(content: Any) -> str:
    """Message content -> text. Handles strings, message objects and lists of
    content blocks ({"type": "text", "text": ...})."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")) if block.get("type", "text") == "text" else "")
            else:
                parts.append(_text(getattr(block, "text", None) or getattr(block, "content", None)))
        return "".join(parts)
    if hasattr(content, "content"):
        return _text(content.content)
    return str(content)


def _args_dict(value: Any) -> dict:
    """Tool arguments in any shape -> dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except ValueError:
            return {"input": value} if value else {}
    if value is None:
        return {}
    return {"input": value}


def _json_safe(meta: Any) -> dict:
    if not isinstance(meta, dict):
        return {}
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
    return out


def _elapsed(t0: float) -> float:
    return round(time.time() - t0, 3)


class _Usage:
    """Token accumulator that remembers whether any usage was reported at all."""

    def __init__(self):
        self.pt = self.ct = 0
        self.seen = False

    def add(self, prompt: Any, completion: Any) -> None:
        if prompt is None and completion is None:
            return
        self.seen = True
        self.pt += int(prompt or 0)
        self.ct += int(completion or 0)

    @property
    def prompt(self):
        return self.pt if self.seen else None

    @property
    def completion(self):
        return self.ct if self.seen else None


def _field(obj: Any, *names: str) -> Any:
    """First present attribute or dict key."""
    for n in names:
        if isinstance(obj, dict) and obj.get(n) is not None:
            return obj[n]
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


class _RetrieverHookMixin:
    """set_retriever only when the config gives a retriever_hook (see python_agent)."""
    hook: Callable | None = None

    @property
    def supports_retriever_injection(self) -> bool:
        return self.hook is not None

    def set_retriever(self, retriever):
        if self.hook is None:
            raise NotImplementedError("set 'retriever_hook: module:function' in the config to enable injection tests")
        self.hook(retriever)


# ---------- LangChain / LangGraph ----------
class _LangChainCollector:
    """Callback methods that record tools, retrieval and token usage. Mixed in
    front of langchain_core's BaseCallbackHandler when that is importable."""

    raise_error = False
    run_inline = True
    ignore_llm = ignore_chain = ignore_agent = ignore_retriever = ignore_chat_model = ignore_retry = False
    ignore_custom_event = False

    def __init__(self):
        self.trace: list[ToolCall] = []
        self.retrieved: list[RetrievedDoc] = []
        self.usage = _Usage()
        self._open: dict[Any, ToolCall] = {}
        self._lock = threading.Lock()

    def on_tool_start(self, serialized, input_str, *, run_id=None, inputs=None, **kw):
        name = (serialized or {}).get("name") or kw.get("name") or "tool"
        call = ToolCall(tool=str(name), args=_args_dict(inputs if inputs is not None else input_str))
        with self._lock:
            self.trace.append(call)
            self._open[run_id] = call

    def on_tool_end(self, output, *, run_id=None, **kw):
        with self._lock:
            call = self._open.pop(run_id, None)
        if call is not None:
            call.result = _text(output)[:_TEXT_LIMIT]

    def on_tool_error(self, error, *, run_id=None, **kw):
        with self._lock:
            call = self._open.pop(run_id, None)
        if call is not None:
            call.result = f"Error: {type(error).__name__}: {error}"

    def on_llm_end(self, response, **kw):
        per_message = []
        for gens in getattr(response, "generations", None) or []:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None)
                if um:
                    per_message.append(um)
        with self._lock:
            if per_message:  # newer LangChain: usage on the AIMessage
                for um in per_message:
                    self.usage.add(_field(um, "input_tokens"), _field(um, "output_tokens"))
                return
            tu = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            self.usage.add(_field(tu, "prompt_tokens", "input_tokens"),
                           _field(tu, "completion_tokens", "output_tokens"))

    def on_retriever_end(self, documents, *, run_id=None, **kw):
        docs = [_lc_doc(d, i) for i, d in enumerate(documents or [])]
        with self._lock:
            self.retrieved += docs


def _lc_doc(d: Any, i: int) -> RetrievedDoc:
    meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {}) or {}
    text = getattr(d, "page_content", None)
    if text is None and isinstance(d, dict):
        text = d.get("page_content") or d.get("text")
    doc_id = getattr(d, "id", None) or meta.get("id") or meta.get("source") or meta.get("file_name") or f"doc{i}"
    score = meta.get("score", meta.get("relevance_score"))
    return RetrievedDoc(id=str(doc_id), text=str(text or "")[:_TEXT_LIMIT],
                        score=float(score) if isinstance(score, (int, float)) else None,
                        metadata=_json_safe(meta))


def _make_lc_handler() -> _LangChainCollector:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:  # fakes / tests: the duck-typed collector is enough
        return _LangChainCollector()
    cls = type("EvalkitCallbackHandler", (_LangChainCollector, BaseCallbackHandler), {})
    return cls()


class LangChainAgent(_RetrieverHookMixin, AgentAdapter):
    """Any LangChain Runnable: AgentExecutor, compiled LangGraph graph, LCEL chain."""

    INPUT_FORMATS = ("dict", "messages", "string")

    def __init__(self, target: Any, input_key: str = "input", output_key: str | None = None,
                 input_format: str = "dict", history_key: str | None = None, inputs: dict | None = None,
                 retriever_hook: Any = None, config: dict | None = None, **_):
        if input_format not in self.INPUT_FORMATS:
            raise ConfigError(f"langchain: input_format must be one of {list(self.INPUT_FORMATS)}, got {input_format!r}")
        self.runnable = _resolve_target(target, ("invoke",), "langchain")
        self.input_key, self.output_key, self.input_format = input_key, output_key, input_format
        self.history_key = history_key
        self.inputs = dict(inputs or {})
        self.config = dict(config or {})
        self.hook = _optional_object(retriever_hook)

    def _payload(self, question: str, history: list[tuple[str, str]]) -> Any:
        if self.input_format == "string":
            return question
        if self.input_format == "messages":
            msgs = []
            for u, a in history:
                msgs += [("user", u), ("assistant", a)]
            return {**self.inputs, "messages": msgs + [("user", question)]}
        payload = {**self.inputs, self.input_key: question}
        if self.history_key:
            payload[self.history_key] = [m for u, a in history for m in (("human", u), ("ai", a))]
        return payload

    def _answer(self, out: Any) -> str:
        if self.output_key:
            return _text(get_path(out, self.output_key, "") if isinstance(out, (dict, list)) else
                         getattr(out, self.output_key, ""))
        if isinstance(out, dict):
            if out.get("messages"):
                return _text(out["messages"][-1])
            for k in ("output", "answer", "result", "text", "response"):
                if k in out:
                    return _text(out[k])
            return json.dumps(out, default=str)[:_TEXT_LIMIT]
        return _text(out)

    @staticmethod
    def _output_docs(out: Any) -> list[RetrievedDoc]:
        """RetrievalQA (return_source_documents) and create_retrieval_chain return docs in the output."""
        if not isinstance(out, dict):
            return []
        docs = out.get("source_documents") or out.get("context")
        if isinstance(docs, list) and all(hasattr(d, "page_content") for d in docs):
            return [_lc_doc(d, i) for i, d in enumerate(docs)]
        return []

    def _invoke(self, question: str, history: list[tuple[str, str]]) -> AgentResult:
        t0 = time.time()
        handler = _make_lc_handler()
        config = {**self.config, "callbacks": list(self.config.get("callbacks") or []) + [handler]}
        try:
            out = self.runnable.invoke(self._payload(question, history), config=config)
        except Exception as e:
            return AgentResult(answer="", trace=handler.trace, retrieved=handler.retrieved,
                               latency_s=_elapsed(t0), error=f"{type(e).__name__}: {e}")
        return AgentResult(answer=self._answer(out).strip(), trace=handler.trace,
                           retrieved=handler.retrieved or self._output_docs(out),
                           prompt_tokens=handler.usage.prompt, completion_tokens=handler.usage.completion,
                           latency_s=_elapsed(t0))

    def run(self, input: str) -> AgentResult:
        return self._invoke(input, [])

    def run_conversation(self, turns: list[str]) -> AgentResult:
        """Native history when the runnable takes it (message graphs, or a history_key)."""
        if self.input_format != "messages" and not self.history_key:
            return super().run_conversation(turns)
        return _conversation(turns, lambda turn, history: self._invoke(turn, history))

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"loaded {type(self.runnable).__name__} (call not probed)")


def _conversation(turns: list[str], step: Callable[[str, list[tuple[str, str]]], AgentResult],
                  t0: float | None = None) -> AgentResult:
    """Drive a natively multi-turn system and merge per-turn results."""
    t0 = time.time() if t0 is None else t0
    history: list[tuple[str, str]] = []
    trace, retrieved, usage = [], [], _Usage()
    last = AgentResult(answer="")
    for turn in turns:
        last = step(turn, list(history))
        history.append((turn, last.answer))
        trace += last.trace
        retrieved += last.retrieved
        usage.add(last.prompt_tokens, last.completion_tokens)
        if last.error:
            break
    return AgentResult(answer=last.answer, trace=trace, retrieved=retrieved, prompt_tokens=usage.prompt,
                       completion_tokens=usage.completion, latency_s=_elapsed(t0), error=last.error,
                       status=last.status, raw={"turns": [{"user": u, "assistant": a} for u, a in history]})


# ---------- LlamaIndex ----------
def _li_node(n: Any, i: int) -> RetrievedDoc:
    node = getattr(n, "node", n)
    meta = getattr(node, "metadata", None) or {}
    get_content = getattr(node, "get_content", None)
    text = get_content() if callable(get_content) else (getattr(node, "text", None) or "")
    doc_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or meta.get("file_name") or f"node{i}"
    score = getattr(n, "score", None)
    return RetrievedDoc(id=str(doc_id), text=str(text)[:_TEXT_LIMIT],
                        score=float(score) if isinstance(score, (int, float)) else None,
                        metadata=_json_safe(meta))


def _li_tool(t: Any) -> ToolCall:
    raw = getattr(t, "raw_input", None)
    if isinstance(raw, dict) and set(raw) <= {"args", "kwargs"}:  # LlamaIndex stores {"args": (...), "kwargs": {...}}
        args = dict(raw.get("kwargs") or {})
        if raw.get("args"):
            args["args"] = list(raw["args"])
    else:
        args = _args_dict(raw)
    content = getattr(t, "content", None)
    if content is None:
        content = getattr(t, "raw_output", "")
    return ToolCall(tool=str(getattr(t, "tool_name", "tool")), args=args, result=str(content)[:_TEXT_LIMIT])


class LlamaIndexAgent(AgentAdapter):
    """A LlamaIndex QueryEngine, ChatEngine, or agent with .query()/.chat()."""

    def __init__(self, target: Any, mode: str = "auto", token_counter: Any = None, **_):
        if mode not in ("auto", "query", "chat"):
            raise ConfigError(f"llamaindex: mode must be auto, query or chat, got {mode!r}")
        self.engine = _resolve_target(target, ("query", "chat"), "llamaindex")
        if mode == "auto":
            mode = "query" if hasattr(self.engine, "query") else "chat"
        if not hasattr(self.engine, mode):
            raise ConfigError(f"llamaindex: mode {mode!r} but {type(self.engine).__name__} has no .{mode}()")
        self.mode = mode
        # A TokenCountingHandler the user attached to Settings.callback_manager (or the engine).
        self.token_counter = _optional_object(token_counter)

    def _call(self, fn: Callable[[], Any]) -> AgentResult:
        t0 = time.time()
        tc = self.token_counter
        if tc is not None and hasattr(tc, "reset_counts"):
            tc.reset_counts()
        try:
            resp = fn()
        except Exception as e:
            return AgentResult(answer="", latency_s=_elapsed(t0), error=f"{type(e).__name__}: {e}")
        answer = getattr(resp, "response", None)
        answer = str(resp) if answer is None else str(answer)
        nodes = getattr(resp, "source_nodes", None) or []
        tools = getattr(resp, "sources", None) or []
        pt = getattr(tc, "prompt_llm_token_count", None) if tc is not None else None
        ct = getattr(tc, "completion_llm_token_count", None) if tc is not None else None
        return AgentResult(answer=answer.strip(), trace=[_li_tool(t) for t in tools],
                           retrieved=[_li_node(n, i) for i, n in enumerate(nodes)],
                           prompt_tokens=pt, completion_tokens=ct, latency_s=_elapsed(t0))

    def run(self, input: str) -> AgentResult:
        if self.mode == "chat":
            self._reset()  # one test case = one fresh conversation
            return self._call(lambda: self.engine.chat(input))
        return self._call(lambda: self.engine.query(input))

    def _reset(self) -> None:
        reset = getattr(self.engine, "reset", None)
        if callable(reset):
            reset()

    def run_conversation(self, turns: list[str]) -> AgentResult:
        """Chat engines and agents keep their own memory: send each turn as is."""
        if not hasattr(self.engine, "chat"):
            return super().run_conversation(turns)
        self._reset()
        return _conversation(turns, lambda turn, _h: self._call(lambda: self.engine.chat(turn)))

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"loaded {type(self.engine).__name__} (mode {self.mode}; call not probed)")


# ---------- CrewAI ----------
class CrewAIAgent(AgentAdapter):
    """A CrewAI Crew: kickoff(inputs={input_key: question})."""

    def __init__(self, target: Any, input_key: str = "input", inputs: dict | None = None,
                 fresh_copy: bool = True, **_):
        self.crew = _resolve_target(target, ("kickoff",), "crewai")
        self.input_key = input_key
        self.inputs = dict(inputs or {})
        self.fresh_copy = fresh_copy

    def run(self, input: str) -> AgentResult:
        t0 = time.time()
        # A copy per case keeps crew memory and task outputs from leaking between test cases.
        crew = self.crew.copy() if self.fresh_copy and callable(getattr(self.crew, "copy", None)) else self.crew
        try:
            result = crew.kickoff(inputs={**self.inputs, self.input_key: input})
        except Exception as e:
            return AgentResult(answer="", latency_s=_elapsed(t0), error=f"{type(e).__name__}: {e}")
        raw = getattr(result, "raw", None)
        answer = str(raw) if raw else str(result)
        usage = getattr(result, "token_usage", None)
        trace = []
        for t in getattr(result, "tasks_output", None) or []:
            agent = getattr(t, "agent", None)
            agent = getattr(agent, "role", agent)
            trace.append(ToolCall(tool=str(agent or getattr(t, "name", None) or "task"),
                                  args={"description": str(getattr(t, "description", "") or "")},
                                  result=str(getattr(t, "raw", "") or "")[:_TEXT_LIMIT]))
        total = _field(usage, "total_tokens") if usage is not None else None
        return AgentResult(answer=answer.strip(), trace=trace,
                           prompt_tokens=_field(usage, "prompt_tokens") if usage is not None else None,
                           completion_tokens=_field(usage, "completion_tokens") if usage is not None else None,
                           latency_s=_elapsed(t0), raw={"total_tokens": total} if total is not None else {})

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"loaded {type(self.crew).__name__} (kickoff not probed)")


# ---------- Semantic Kernel ----------
class _AsyncRunner:
    """Runs coroutines on one private event loop in a worker thread. Works when
    the caller already has a running loop (Jupyter, async apps), and keeps async
    HTTP clients bound to a single loop across test cases."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def run(self, coro_factory: Callable[[], Any]) -> Any:
        with self._lock:
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                self.thread = threading.Thread(target=self.loop.run_forever, name="evalkit-async", daemon=True)
                self.thread.start()
        return asyncio.run_coroutine_threadsafe(coro_factory(), self.loop).result()

    def close(self) -> None:
        with self._lock:
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(timeout=5)
                self.loop.close()
                self.loop = self.thread = None


def _sk_kind(item: Any) -> str | None:
    kind = getattr(item, "content_type", None)
    kind = getattr(kind, "value", kind)
    if kind in ("function_call", "function_result"):
        return kind
    name = type(item).__name__
    return {"FunctionCallContent": "function_call", "FunctionResultContent": "function_result"}.get(name)


class _SkCollector:
    """Pairs FunctionCallContent with FunctionResultContent by call id and sums usage."""

    def __init__(self):
        self.trace: list[ToolCall] = []
        self.usage = _Usage()
        self._by_id: dict[str, ToolCall] = {}
        self._seen: list[Any] = []  # strong refs: id() alone can be reused after garbage collection

    def message(self, msg: Any) -> None:
        if msg is None or any(m is msg for m in self._seen):
            return
        self._seen.append(msg)
        items = getattr(msg, "items", None)
        for item in items if isinstance(items, (list, tuple)) else []:
            kind = _sk_kind(item)
            if kind == "function_call":
                name = _sk_name(item)
                call = ToolCall(tool=name, args=_args_dict(getattr(item, "arguments", None)))
                self.trace.append(call)
                if getattr(item, "id", None):
                    self._by_id[item.id] = call
            elif kind == "function_result":
                result = str(getattr(item, "result", ""))[:_TEXT_LIMIT]
                call = self._by_id.get(getattr(item, "id", None) or getattr(item, "call_id", None) or "")
                if call is None:
                    call = ToolCall(tool=_sk_name(item))
                    self.trace.append(call)
                call.result = result
        self.metadata(getattr(msg, "metadata", None))

    def metadata(self, meta: Any) -> None:
        usage = meta.get("usage") if isinstance(meta, dict) else None
        if usage is not None:
            self.usage.add(_field(usage, "prompt_tokens", "input_tokens"),
                           _field(usage, "completion_tokens", "output_tokens"))


def _sk_name(item: Any) -> str:
    fn = getattr(item, "function_name", None) or getattr(item, "name", None) or "function"
    plugin = getattr(item, "plugin_name", None)
    return f"{plugin}-{fn}" if plugin and not str(fn).startswith(f"{plugin}-") else str(fn)


class SemanticKernelAgent(AgentAdapter):
    """A Semantic Kernel Kernel + function, or an SK agent (ChatCompletionAgent)."""

    def __init__(self, target: Any, function: str | None = None, plugin_name: str | None = None,
                 function_name: str | None = None, input_arg: str = "input", **_):
        self.obj = _resolve_target(target, ("invoke", "get_response"), "semantic_kernel")
        if function:
            plugin_name, _, function_name = function.rpartition(".")
        self.plugin_name, self.function_name = plugin_name or None, function_name
        self.is_agent = not function_name and (hasattr(self.obj, "get_response") or hasattr(self.obj, "instructions"))
        if not self.is_agent and not function_name:
            raise ConfigError("semantic_kernel: target looks like a Kernel; set function: plugin_name.function_name "
                              "(or target an SK agent such as ChatCompletionAgent)")
        self.input_arg = input_arg
        self._runner = _AsyncRunner()

    # --- kernel function ---
    async def _invoke_function(self, question: str):
        return await self.obj.invoke(plugin_name=self.plugin_name, function_name=self.function_name,
                                     **{self.input_arg: question})

    def _function_result(self, res: Any, col: _SkCollector) -> str:
        value = getattr(res, "value", res)
        msgs = value if isinstance(value, list) else [value]
        for m in msgs:
            col.message(m)
        col.metadata(getattr(res, "metadata", None))
        if isinstance(value, list) and value:
            return _text(value[-1])
        return _text(value) if value is not None else str(res)

    # --- agent ---
    async def _invoke_agent(self, question: str, thread: Any, col: _SkCollector):
        async def on_intermediate(message):
            col.message(message)

        if not hasattr(self.obj, "invoke"):  # agent exposing only get_response
            item = await self.obj.get_response(messages=question, thread=thread)
            msg = getattr(item, "message", item)
            col.message(msg)
            return msg, getattr(item, "thread", None) or thread
        last, new_thread = None, thread
        try:
            gen = self.obj.invoke(messages=question, thread=thread, on_intermediate_message=on_intermediate)
        except TypeError:  # older SK: no intermediate-message callback
            gen = self.obj.invoke(messages=question, thread=thread)
        async for item in gen:
            msg = getattr(item, "message", item)  # AgentResponseItem wraps ChatMessageContent
            new_thread = getattr(item, "thread", None) or new_thread
            col.message(msg)
            last = msg
        return last, new_thread

    def _turn(self, question: str, thread: Any = None) -> tuple[AgentResult, Any]:
        t0, col = time.time(), _SkCollector()
        try:
            if self.is_agent:
                msg, thread = self._runner.run(lambda: self._invoke_agent(question, thread, col))
                answer = _text(msg)
            else:
                answer = self._function_result(self._runner.run(lambda: self._invoke_function(question)), col)
        except Exception as e:
            return AgentResult(answer="", trace=col.trace, latency_s=_elapsed(t0),
                               error=f"{type(e).__name__}: {e}"), thread
        return AgentResult(answer=answer.strip(), trace=col.trace, prompt_tokens=col.usage.prompt,
                           completion_tokens=col.usage.completion, latency_s=_elapsed(t0)), thread

    def run(self, input: str) -> AgentResult:
        return self._turn(input)[0]

    def run_conversation(self, turns: list[str]) -> AgentResult:
        """Agents keep history in an AgentThread, passed from turn to turn."""
        if not self.is_agent:
            return super().run_conversation(turns)
        state = {"thread": None}

        def step(turn, _history):
            result, state["thread"] = self._turn(turn, state["thread"])
            return result
        return _conversation(turns, step)

    def health_check(self) -> HealthStatus:
        what = "agent" if self.is_agent else f"function {self.plugin_name}.{self.function_name}"
        return timed_health(lambda: f"loaded {type(self.obj).__name__} {what} (call not probed)")

    def close(self) -> None:
        self._runner.close()
