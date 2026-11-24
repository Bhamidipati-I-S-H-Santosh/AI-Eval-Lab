"""Connect any agent that has an HTTP API, by configuration only.

config:
  type: http
  url: https://bot.example.com/api/chat
  method: POST
  headers: { Authorization: "Bearer ${BOT_API_KEY}" }   # ${VAR} read from env/.env
  request: { message: "{input}", session_id: "eval" }   # {input} = the test question
  answer_field: reply.text                              # dotted path in the JSON response
  trace_field: debug.tool_calls                         # optional
  trace_map: { tool: name, args: arguments, result: output }   # optional key names
  retrieved_field: debug.sources                        # optional
  retrieved_map: { id: doc_id, text: content }          # optional key names
  usage_fields: { prompt_tokens: usage.input_tokens, completion_tokens: usage.output_tokens }
  timeout: 60
  retries: 2
"""
from __future__ import annotations

import json
import time

import requests

from ..types import AgentResult, RetrievedDoc, ToolCall
from ..utils import expand_env, fill_template, get_path
from .base import AgentAdapter


class HttpAgentAdapter(AgentAdapter):
    def __init__(self, url: str, method: str = "POST", headers: dict | None = None,
                 request: dict | str | None = None, answer_field: str = "answer",
                 trace_field: str | None = None, trace_map: dict | None = None,
                 retrieved_field: str | None = None, retrieved_map: dict | None = None,
                 usage_fields: dict | None = None, timeout: float = 60, retries: int = 2, **_):
        self.url = expand_env(url)
        self.method = method.upper()
        self.headers = expand_env(headers or {})
        self.request = request if request is not None else {"input": "{input}"}
        self.answer_field = answer_field
        self.trace_field, self.trace_map = trace_field, {"tool": "tool", "args": "args",
                                                          "result": "result", **(trace_map or {})}
        self.retrieved_field, self.retrieved_map = retrieved_field, {"id": "id", "text": "text",
                                                                      "score": "score", **(retrieved_map or {})}
        self.usage_fields = usage_fields or {}
        self.timeout, self.retries = timeout, retries
        self.session = requests.Session()

    def _call(self, input: str) -> dict:
        body = fill_template(self.request, {"input": input, "question": input})
        last = None
        for attempt in range(self.retries + 1):
            try:
                if self.method == "GET":
                    r = self.session.get(self.url, params=body, headers=self.headers, timeout=self.timeout)
                else:
                    r = self.session.request(self.method, self.url, json=body,
                                             headers=self.headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                try:
                    return r.json()
                except json.JSONDecodeError:
                    return {"answer": r.text}
            except requests.RequestException as e:
                last = e
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"HTTP call failed: {last}")

    def run(self, input: str) -> AgentResult:
        t0 = time.time()
        data = self._call(input)
        tm, rm = self.trace_map, self.retrieved_map
        trace = [ToolCall(tool=str(get_path(t, tm["tool"], "")),
                          args=get_path(t, tm["args"], {}) or {},
                          result=str(get_path(t, tm["result"], "")))
                 for t in (get_path(data, self.trace_field, []) or [])]
        retrieved = [RetrievedDoc(id=str(get_path(d, rm["id"], "")), text=str(get_path(d, rm["text"], "")),
                                  score=get_path(d, rm["score"]))
                     for d in (get_path(data, self.retrieved_field, []) or [])]
        return AgentResult(
            answer=str(get_path(data, self.answer_field, "") or ""),
            trace=trace, retrieved=retrieved,
            prompt_tokens=get_path(data, self.usage_fields.get("prompt_tokens")),
            completion_tokens=get_path(data, self.usage_fields.get("completion_tokens")),
            latency_s=round(time.time() - t0, 3), raw=data)

    def close(self):
        self.session.close()
