"""Google Vertex AI agents over REST: Agent Engine (reasoning engines) and
Dialogflow CX / Vertex AI Agent Builder conversational agents.

config (Agent Engine, ADK agent - the default template):
  type: vertex_agent
  mode: agent_engine
  project: my-gcp-project
  location: us-central1
  reasoning_engine_id: "1234567890123456789"    # or the full projects/.../reasoningEngines/... name
  engine_api: adk          # adk: :streamQuery class_method stream_query (sessions via create_session)
                           # query: :query with {"input": {<input_key>: question}} (LangChain/LangGraph/custom)
  user_id: evalkit
  auth:                    # optional; default: Application Default Credentials (cloud-platform scope)
    type: gcp_service_account

config (Dialogflow CX / Agent Builder conversational agent):
  type: vertex_agent
  mode: dialogflow_cx
  project: my-gcp-project
  location: global                              # or e.g. us-central1 / europe-west1
  agent_id: 0b1c2d3e-...                        # the agent's UUID
  environment: null                             # optional environment id (default: draft)
  language_code: en

APIs: aiplatform.googleapis.com v1 reasoningEngines.query / streamQuery and
dialogflow.googleapis.com v3 projects.locations.agents.sessions.detectIntent.
Needs: pip install 'evalkit[gcp]' (for google-auth), unless a pre-authorized
session (e.g. google.auth.transport.requests.AuthorizedSession) is injected.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from ...auth import build_auth
from ...errors import ConfigError, PluginDependencyError
from ...plugins import HealthStatus, timed_health
from ...types import AgentResult, RetrievedDoc, ToolCall
from ..base import AgentAdapter

_PLUGIN = "system:vertex_agent"
_MAX = 4000
_MODES = ("agent_engine", "dialogflow_cx")


def _k(d: dict, snake: str, default: Any = None) -> Any:
    """Read a key that may be serialized as snake_case or camelCase."""
    if not isinstance(d, dict):
        return default
    if snake in d and d[snake] is not None:
        return d[snake]
    camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), snake)
    v = d.get(camel)
    return default if v is None else v


def _dump(v: Any) -> str:
    if v is None:
        return ""
    return (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str))[:_MAX]


def _add(docs: list[RetrievedDoc], doc: RetrievedDoc) -> None:
    for d in docs:
        if d.id == doc.id:
            d.metadata["cited"] = d.metadata.get("cited") or doc.metadata.get("cited")
            d.text = d.text or doc.text
            return
    docs.append(doc)


class VertexAgent(AgentAdapter):
    def __init__(self, project: str | None = None, location: str = "us-central1", mode: str = "agent_engine",
                 reasoning_engine_id: str | None = None, engine_api: str = "adk",
                 query_method: str | None = None, create_session_method: str = "create_session",
                 input_key: str = "input", extra_input: dict | None = None, user_id: str = "evalkit",
                 agent_id: str | None = None, environment: str | None = None, language_code: str = "en",
                 query_params: dict | None = None, api_version: str = "v1",
                 auth: dict | None = None, timeout: float = 120, session=None, **_):
        if mode not in _MODES:
            raise ConfigError(f"vertex_agent: mode must be one of {list(_MODES)}, got '{mode}'")
        self.mode, self.location, self.timeout = mode, location or "us-central1", float(timeout)
        self.user_id, self.language_code = user_id, language_code
        self.query_params, self.extra_input, self.input_key = dict(query_params or {}), dict(extra_input or {}), input_key
        if mode == "agent_engine":
            if not reasoning_engine_id:
                raise ConfigError("vertex_agent (agent_engine) needs 'reasoning_engine_id' "
                                  "(the number at the end of the Agent Engine resource name)")
            m = re.match(r"projects/([^/]+)/locations/([^/]+)/reasoningEngines/([^/]+)$", str(reasoning_engine_id))
            if m:
                project, self.location, reasoning_engine_id = m.groups()
            if engine_api not in ("adk", "query"):
                raise ConfigError("vertex_agent: engine_api must be 'adk' or 'query'")
            if not project:
                raise ConfigError("vertex_agent needs 'project' (GCP project id)")
            self.engine_api = engine_api
            self.query_method = query_method or ("stream_query" if engine_api == "adk" else "query")
            self.create_session_method = create_session_method
            self.resource = (f"https://{self.location}-aiplatform.googleapis.com/{api_version}/projects/{project}"
                             f"/locations/{self.location}/reasoningEngines/{reasoning_engine_id}")
        else:
            if not agent_id:
                raise ConfigError("vertex_agent (dialogflow_cx) needs 'agent_id' (the agent UUID)")
            m = re.match(r"projects/([^/]+)/locations/([^/]+)/agents/([^/]+)$", str(agent_id))
            if m:
                project, self.location, agent_id = m.groups()
            if not project:
                raise ConfigError("vertex_agent needs 'project' (GCP project id)")
            host = ("dialogflow.googleapis.com" if self.location == "global"
                    else f"{self.location}-dialogflow.googleapis.com")
            self.resource = f"https://{host}/v3/projects/{project}/locations/{self.location}/agents/{agent_id}"
            self.session_base = self.resource + (f"/environments/{environment}" if environment else "") + "/sessions"
        self.project = project
        # Auth: explicit auth block > Google ADC. An injected session without an
        # auth block is assumed to be pre-authorized (e.g. AuthorizedSession).
        if auth:
            self.auth = build_auth(auth)
        elif session is None:
            try:
                import google.auth  # noqa: F401 - fail fast with an install hint
            except ModuleNotFoundError as e:
                raise PluginDependencyError(_PLUGIN, e.name or "google-auth", "gcp") from e
            from ...auth import GcpServiceAccountAuth
            self.auth = GcpServiceAccountAuth()
        else:
            self.auth = None
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    # ---- HTTP ----
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.auth is not None:
            h.update(self.auth.headers())
        return h

    def _request(self, method: str, url: str, body: dict | None = None):
        r = self.session.request(method, url, json=body, headers=self._headers(), timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Vertex {self.mode} {method} {url} failed: HTTP {r.status_code} {r.text[:300]}")
        return r

    # ---- Agent Engine ----
    def _create_session(self) -> str:
        r = self._request("POST", self.resource + ":query",
                          {"class_method": self.create_session_method, "input": {"user_id": self.user_id}})
        out = r.json().get("output")
        sid = out.get("id") if isinstance(out, dict) else out
        if not sid:
            raise RuntimeError(f"Agent Engine {self.create_session_method} returned no session id: {r.text[:200]}")
        return str(sid)

    def _adk_turn(self, text: str, session_id: str | None) -> dict:
        inp: dict[str, Any] = {"user_id": self.user_id, "message": text, **self.extra_input}
        if session_id:
            inp["session_id"] = session_id
        r = self._request("POST", self.resource + ":streamQuery?alt=sse",
                          {"class_method": self.query_method, "input": inp})
        trace: list[ToolCall] = []
        retrieved: list[RetrievedDoc] = []
        pending: dict[str, ToolCall] = {}
        texts: list[str] = []
        pt = ct = 0
        have_tokens = False
        n_events = 0
        for line in r.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            n_events += 1
            if _k(ev, "error_code") or _k(ev, "error_message") or "error" in ev:
                err = ev.get("error") or {}
                msg = _k(ev, "error_message") or (err.get("message") if isinstance(err, dict) else err)
                raise RuntimeError(f"Agent Engine stream error {_k(ev, 'error_code', '')}: {msg}")
            um = _k(ev, "usage_metadata") or {}
            if _k(um, "prompt_token_count") is not None or _k(um, "candidates_token_count") is not None:
                have_tokens = True
                pt += int(_k(um, "prompt_token_count", 0))
                ct += int(_k(um, "candidates_token_count", 0))
            gm = _k(ev, "grounding_metadata") or {}
            for ch in _k(gm, "grounding_chunks", []) or []:
                src = _k(ch, "retrieved_context") or ch.get("web") or {}
                if src.get("uri") or src.get("title"):
                    _add(retrieved, RetrievedDoc(id=str(src.get("uri") or src.get("title")),
                                                 text=str(src.get("text") or src.get("title") or ""),
                                                 metadata={"title": src.get("title"), "cited": True,
                                                           "source": "grounding"}))
            content = ev.get("content") or {}
            from_user = content.get("role") == "user"   # ADK sends function responses with role "user"
            for part in content.get("parts") or []:
                fc, fr = _k(part, "function_call"), _k(part, "function_response")
                if fc:
                    call = ToolCall(str(fc.get("name", "tool")), dict(fc.get("args") or {}))
                    trace.append(call)
                    pending[str(fc.get("id") or fc.get("name"))] = call
                elif fr:
                    call = pending.pop(str(fr.get("id") or fr.get("name")), None) or \
                        next((c for c in reversed(trace) if c.tool == fr.get("name") and not c.result), None)
                    if call is not None:
                        call.result = _dump(fr.get("response"))
                    texts = []           # text before a tool result is intermediate chatter
                elif part.get("text") and not part.get("thought") and not from_user:
                    texts.append(part["text"])
        return {"answer": "".join(texts).strip(), "trace": trace, "retrieved": retrieved,
                "pt": pt if have_tokens else None, "ct": ct if have_tokens else None,
                "raw": {"events": n_events}}

    def _query_turn(self, text: str) -> dict:
        r = self._request("POST", self.resource + ":query",
                          {"class_method": self.query_method, "input": {self.input_key: text, **self.extra_input}})
        data = r.json()
        out = data.get("output", data)
        trace: list[ToolCall] = []
        answer = out
        if isinstance(out, dict):
            answer = next((out[k] for k in ("output", "answer", "content", "text", "response") if out.get(k)), None)
            if answer is None and isinstance(out.get("messages"), list) and out["messages"]:
                last = out["messages"][-1]                     # LangGraph state
                answer = (last.get("content") if isinstance(last, dict) else last) or ""
                if not isinstance(answer, str):
                    answer = (last.get("kwargs") or {}).get("content", "") if isinstance(last, dict) else str(last)
            for step in out.get("intermediate_steps") or []:   # LangChain AgentExecutor
                action, obs = (step + [None, None])[:2] if isinstance(step, list) else (step, None)
                a = action.get("kwargs", action) if isinstance(action, dict) else {}
                tin = a.get("tool_input")
                trace.append(ToolCall(str(a.get("tool", "tool")), tin if isinstance(tin, dict) else {"input": tin},
                                      _dump(obs)))
        if not isinstance(answer, str):
            answer = _dump(answer)
        return {"answer": answer.strip(), "trace": trace, "retrieved": [], "pt": None, "ct": None, "raw": {}}

    # ---- Dialogflow CX ----
    def _cx_turn(self, text: str, session_id: str) -> dict:
        body = {"queryInput": {"text": {"text": text}, "languageCode": self.language_code},
                "queryParams": {"populateDataStoreConnectionSignals": True, **self.query_params}}
        data = self._request("POST", f"{self.session_base}/{session_id}:detectIntent", body).json()
        qr = data.get("queryResult") or {}
        texts = [t for m in qr.get("responseMessages") or [] for t in ((m.get("text") or {}).get("text") or [])]
        trace: list[ToolCall] = []
        actions = ((qr.get("generativeInfo") or {}).get("actionTracingInfo") or {}).get("actions") or []
        for a in actions:
            if a.get("toolUse"):
                tu = a["toolUse"]
                name = tu.get("displayName") or tu.get("action") or str(tu.get("tool", "tool")).rsplit("/", 1)[-1]
                args = dict(tu.get("inputActionParameters") or {})
                if tu.get("action") and tu.get("action") != name:
                    args.setdefault("_action", tu.get("action"))
                trace.append(ToolCall(str(name), args, _dump(tu.get("outputActionParameters"))))
            elif a.get("playbookInvocation"):
                pb = a["playbookInvocation"]
                trace.append(ToolCall("playbook:" + str(pb.get("displayName") or pb.get("playbook", "")).rsplit("/", 1)[-1]))
            elif a.get("flowInvocation"):
                fl = a["flowInvocation"]
                trace.append(ToolCall("flow:" + str(fl.get("displayName") or fl.get("flow", "")).rsplit("/", 1)[-1]))
        retrieved: list[RetrievedDoc] = []
        sig = qr.get("dataStoreConnectionSignals") or {}
        for cs in sig.get("citedSnippets") or []:
            s = cs.get("searchSnippet") or {}
            _add(retrieved, RetrievedDoc(id=str(s.get("documentUri") or s.get("documentTitle") or ""),
                                         text=s.get("text") or "",
                                         metadata={"title": s.get("documentTitle"), "cited": True}))
        for s in sig.get("searchSnippets") or []:
            _add(retrieved, RetrievedDoc(id=str(s.get("documentUri") or s.get("documentTitle") or ""),
                                         text=s.get("text") or "",
                                         metadata={"title": s.get("documentTitle"), "cited": False}))
        raw = {"response_id": data.get("responseId"),
               "intent": (qr.get("intent") or {}).get("displayName") or (qr.get("match") or {}).get("intent", {}).get("displayName"),
               "match_type": (qr.get("match") or {}).get("matchType"),
               "page": (qr.get("currentPage") or {}).get("displayName")}
        return {"answer": "\n".join(texts).strip(), "trace": trace, "retrieved": retrieved,
                "pt": None, "ct": None, "raw": raw}

    # ---- AgentAdapter ----
    def run(self, input: str) -> AgentResult:
        if self.mode == "agent_engine" and self.engine_api == "query":
            t0 = time.monotonic()
            r = self._query_turn(input)
            return AgentResult(answer=r["answer"], trace=r["trace"], latency_s=round(time.monotonic() - t0, 3),
                               raw={"mode": self.mode})
        return self.run_conversation([input])

    def run_conversation(self, turns: list[str]) -> AgentResult:
        if self.mode == "agent_engine" and self.engine_api == "query":
            return super().run_conversation(turns)        # no native session: transcript replay
        t0 = time.monotonic()
        if self.mode == "dialogflow_cx":
            session_id = f"evalkit-{uuid.uuid4().hex}"
        else:
            session_id = self._create_session() if len(turns) > 1 else None
        trace: list[ToolCall] = []
        retrieved: list[RetrievedDoc] = []
        pt = ct = 0
        have_tokens = False
        answer, raw = "", {"mode": self.mode, "session_id": session_id}
        for turn in turns:
            r = self._cx_turn(turn, session_id) if self.mode == "dialogflow_cx" else self._adk_turn(turn, session_id)
            answer = r["answer"]
            trace += r["trace"]
            for d in r["retrieved"]:
                _add(retrieved, d)
            if r["pt"] is not None or r["ct"] is not None:
                have_tokens = True
                pt += r["pt"] or 0
                ct += r["ct"] or 0
            raw.update({k: v for k, v in r["raw"].items() if v is not None})
        if len(turns) > 1:
            raw["turns"] = len(turns)
        return AgentResult(answer=answer, trace=trace, retrieved=retrieved,
                           prompt_tokens=pt if have_tokens else None, completion_tokens=ct if have_tokens else None,
                           latency_s=round(time.monotonic() - t0, 3), raw=raw)

    def health_check(self) -> HealthStatus:
        def probe():
            data = self._request("GET", self.resource).json()
            return f"{self.mode} '{data.get('displayName') or data.get('name', self.resource)}' reachable"
        return timed_health(probe)

    def close(self) -> None:
        self.session.close()
