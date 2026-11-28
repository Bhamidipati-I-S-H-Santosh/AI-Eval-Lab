"""Evaluate an MCP tool server: a generic tool-calling agent loop drives any
OpenAI-compatible model against the server's tools.

config:
  type: mcp
  server:                                 # stdio: evalkit starts the server process
    command: python
    args: [mcp_server.py]
    env: { POLICY_DIR: ./data/docs }      # optional, extra environment variables
    cwd: .                                # optional
  # server: { url: https://tools.example.com/mcp, headers: {Authorization: "Bearer ${MCP_TOKEN}"} }
  # server: { url: https://tools.example.com/sse, transport: sse }
  model: { type: openai_compatible, base_url: https://api.groq.com/openai/v1, api_key: "${GROQ_API_KEY}" }
  model_name: llama-3.3-70b-versatile
  system_prompt: "Answer using the tools. Cite the source file."
  max_steps: 6                            # model calls per question before giving up
  tools: [search_policies, read_document] # optional allow-list (default: all server tools)

One MCP session is opened lazily and kept for the whole run; close() ends it.
Tool calling needs a provider whose `.client` speaks the OpenAI chat
completions API (openai_compatible, azure_openai).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future
from typing import Any

from ..errors import ConfigError
from ..plugins import HealthStatus, timed_health
from ..types import AgentResult, ToolCall
from .base import AgentAdapter

_RESULT_LIMIT = 20_000  # chars of a tool result kept in the trace and sent back to the model


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Attribute or dict key, trying each name (MCP SDK 2.x snake_case, 1.x camelCase)."""
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _content_text(result: Any) -> str:
    """Flatten a CallToolResult's content blocks to text."""
    parts = []
    for c in _get(result, "content", default=[]) or []:
        text = _get(c, "text")
        if text is not None:
            parts.append(str(text))
        else:  # image / audio / resource blocks: describe instead of inlining bytes
            kind = _get(c, "type", default="content")
            uri = _get(_get(c, "resource"), "uri")
            parts.append(f"[{kind}{' ' + str(uri) if uri else ''}]")
    if not parts:
        structured = _get(result, "structured_content", "structuredContent")
        if structured is not None:
            parts.append(json.dumps(structured, default=str))
    return "\n".join(parts)


class _McpSession:
    """One MCP client session living on a private event loop in a worker thread,
    so the synchronous adapter can reuse it across test cases."""

    def __init__(self, server: dict):
        self.server = server
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="evalkit-mcp", daemon=True)
        self.thread.start()
        self.session: Any = None
        self._stop: asyncio.Event | None = None
        self._ready: Future = Future()
        self._done = asyncio.run_coroutine_threadsafe(self._lifecycle(), self.loop)
        try:
            self._ready.result(timeout=float(server.get("connect_timeout", 60)))
        except BaseException:
            self.close()
            raise

    def _transport(self):
        s = self.server
        if s.get("url"):
            headers = s.get("headers") or None
            if s.get("transport") == "sse":
                from mcp.client.sse import sse_client
                return sse_client(s["url"], headers=headers)
            try:  # SDK 2.x
                from mcp.client.streamable_http import streamable_http_client
            except ImportError:  # SDK 1.x
                from mcp.client.streamable_http import streamablehttp_client
                return streamablehttp_client(s["url"], headers=headers)
            http_client = None
            if headers:
                from mcp.shared._httpx_utils import create_mcp_http_client
                http_client = create_mcp_http_client(headers=headers)
            return streamable_http_client(s["url"], http_client=http_client)
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=s["command"], args=[str(a) for a in s.get("args") or []],
                                       env={k: str(v) for k, v in (s.get("env") or {}).items()} or None,
                                       cwd=s.get("cwd"))
        return stdio_client(params)

    async def _lifecycle(self) -> None:
        from mcp import ClientSession
        self._stop = asyncio.Event()
        try:
            async with self._transport() as streams:
                read, write = streams[0], streams[1]  # 1.x streamable HTTP also yields a session-id getter
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self._ready.set_result(True)
                    await self._stop.wait()
        except BaseException as e:
            if not self._ready.done():
                self._ready.set_exception(e)
            else:
                raise
        finally:
            self.session = None

    def call(self, coro_fn, timeout: float | None = None):
        if self.session is None:
            raise RuntimeError("MCP session is closed")
        return asyncio.run_coroutine_threadsafe(coro_fn(self.session), self.loop).result(timeout)

    def close(self) -> None:
        if self._stop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._stop.set)
            try:
                self._done.result(timeout=10)
            except BaseException:
                pass
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        if not self.loop.is_running():
            self.loop.close()


class McpToolAgent(AgentAdapter):
    def __init__(self, server: dict, model: dict | Any, model_name: str | None = None,
                 system_prompt: str | None = None, max_steps: int = 6, tools: list[str] | None = None,
                 temperature: float = 0.0, tool_timeout: float = 120, **_):
        if not isinstance(server, dict) or not (server.get("command") or server.get("url")):
            raise ConfigError("mcp: 'server' needs either command (+ args) for a stdio server, "
                              "or url for a streamable HTTP / SSE server")
        if int(max_steps) < 1:
            raise ConfigError("mcp: max_steps must be at least 1")
        self.server = dict(server)
        self.provider = self._build_provider(model)
        self.client = self._chat_client(self.provider)
        if not model_name and isinstance(model, dict):
            model_name = model.get("model_name") or model.get("model")
        self.model_name = model_name or getattr(self.provider, "default_model", None)
        if not self.model_name:
            raise ConfigError("mcp: set 'model_name' (the model id or Azure deployment to drive the tools)")
        self.system_prompt = system_prompt
        self.max_steps, self.temperature, self.tool_timeout = int(max_steps), temperature, tool_timeout
        self.allow = set(tools) if tools else None
        self._session: _McpSession | None = None
        self._tools: list[dict] | None = None
        self._lock = threading.Lock()

    # ---------- setup ----------
    @staticmethod
    def _build_provider(model):
        if isinstance(model, dict):
            from ..plugins import REGISTRY
            cfg = {k: v for k, v in model.items() if k not in ("model_name", "model")}
            cfg.setdefault("type", "openai_compatible")
            return REGISTRY.build("model", cfg)
        return model  # an already-built provider object

    @staticmethod
    def _chat_client(provider):
        client = getattr(provider, "client", None)
        create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
        if not callable(create):
            raise ConfigError(f"mcp: model provider {type(provider).__name__} has no native tool calling. "
                              "Use type: openai_compatible (OpenAI, Groq, vLLM, Ollama /v1, LiteLLM...) "
                              "or type: azure_openai for the model that drives the MCP tools")
        return client

    def _connect(self) -> _McpSession:
        with self._lock:
            if self._session is None:
                self._session = _McpSession(self.server)
            return self._session

    def tool_specs(self) -> list[dict]:
        """The server's tools in OpenAI function-calling format."""
        if self._tools is None:
            listed = self._connect().call(lambda s: s.list_tools(), timeout=self.tool_timeout)
            specs = []
            for t in _get(listed, "tools", default=[]):
                if self.allow is not None and t.name not in self.allow:
                    continue
                schema = _get(t, "input_schema", "inputSchema") or {"type": "object", "properties": {}}
                specs.append({"type": "function", "function": {
                    "name": t.name, "description": t.description or "", "parameters": schema}})
            if not specs:
                raise ConfigError(f"mcp: the server exposes no usable tools (allow-list: {sorted(self.allow or [])})")
            self._tools = specs
        return self._tools

    def call_tool(self, name: str, args: dict) -> tuple[str, bool]:
        res = self._connect().call(lambda s: s.call_tool(name, args), timeout=self.tool_timeout)
        return _content_text(res)[:_RESULT_LIMIT], bool(_get(res, "is_error", "isError", default=False))

    # ---------- agent loop ----------
    def _loop(self, messages: list[dict], acc: dict) -> str:
        tools = self.tool_specs()
        names = {t["function"]["name"] for t in tools}
        for _step in range(self.max_steps):
            resp = self.client.chat.completions.create(model=self.model_name, messages=messages, tools=tools,
                                                       temperature=self.temperature)
            u = getattr(resp, "usage", None)
            if u is not None:
                acc["have_tokens"] = True
                acc["pt"] += getattr(u, "prompt_tokens", 0) or 0
                acc["ct"] += getattr(u, "completion_tokens", 0) or 0
            msg = resp.choices[0].message
            calls = list(getattr(msg, "tool_calls", None) or [])
            if not calls:
                answer = (msg.content or "").strip()
                messages.append({"role": "assistant", "content": answer})
                return answer
            messages.append({"role": "assistant", "content": msg.content or None, "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                for c in calls]})
            for c in calls:
                name = c.function.name
                try:
                    args = json.loads(c.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as e:
                    args, out, is_err = {"_raw": c.function.arguments}, f"Error: invalid tool arguments ({e})", True
                else:
                    if name not in names:
                        out, is_err = f"Error: unknown tool '{name}'", True
                    else:
                        try:
                            out, is_err = self.call_tool(name, args)
                        except Exception as e:  # tool failure goes back to the model, like a real agent
                            out, is_err = f"Error: {type(e).__name__}: {e}", True
                acc["trace"].append(ToolCall(tool=name, args=args, result=out))
                if is_err:
                    acc["tool_errors"] += 1
                messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
        acc["error"] = f"no final answer after max_steps={self.max_steps} model calls"
        return ""

    def _start(self, input: str | None = None) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        if input is not None:
            msgs.append({"role": "user", "content": input})
        return msgs

    def _result(self, answer: str, acc: dict, t0: float, raw: dict) -> AgentResult:
        return AgentResult(answer=answer, trace=acc["trace"],
                           prompt_tokens=acc["pt"] if acc["have_tokens"] else None,
                           completion_tokens=acc["ct"] if acc["have_tokens"] else None,
                           latency_s=round(time.time() - t0, 3), error=acc["error"],
                           raw={**raw, "steps_tool_calls": len(acc["trace"]), "tool_errors": acc["tool_errors"]})

    @staticmethod
    def _acc() -> dict:
        return {"trace": [], "pt": 0, "ct": 0, "have_tokens": False, "error": None, "tool_errors": 0}

    def run(self, input: str) -> AgentResult:
        t0, acc = time.time(), self._acc()
        try:
            answer = self._loop(self._start(input), acc)
        except ConfigError:
            raise
        except Exception as e:
            answer, acc["error"] = "", f"{type(e).__name__}: {e}"
        return self._result(answer, acc, t0, {})

    def run_conversation(self, turns: list[str]) -> AgentResult:
        """Native multi-turn: the full message history (including tool calls) carries over."""
        t0, acc = time.time(), self._acc()
        messages, history, answer = self._start(), [], ""
        for turn in turns:
            messages.append({"role": "user", "content": turn})
            try:
                answer = self._loop(messages, acc)
            except ConfigError:
                raise
            except Exception as e:
                answer, acc["error"] = "", f"{type(e).__name__}: {e}"
            history.append({"user": turn, "assistant": answer})
            if acc["error"]:
                break
        return self._result(answer, acc, t0, {"turns": history})

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"MCP server lists {len(self.tool_specs())} tool(s)")

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None
            self._tools = None
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()
