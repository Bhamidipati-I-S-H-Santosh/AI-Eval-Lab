"""AWS Bedrock Agents via bedrock-agent-runtime InvokeAgent (boto3).

config:
  type: bedrock_agent
  agent_id: ABCDEFGHIJ
  agent_alias_id: TSTALIASID           # or a published alias id
  region: us-east-1
  profile: null                        # optional named AWS profile; default: standard credential chain
  session_attributes: { tenant: northwind }     # optional sessionState.sessionAttributes
  prompt_session_attributes: {}                 # optional sessionState.promptSessionAttributes
  memory_id: null                      # optional, for agents with memory enabled
  timeout: 120                         # max seconds to read one agent response

Each test case gets a fresh sessionId; a multi-turn case reuses one sessionId
for all turns. enableTrace is on, so the trace stream gives action-group,
knowledge-base, code-interpreter and collaborator calls plus token usage;
chunk attributions (citations) and knowledge-base lookups become `retrieved`.
Return-of-control action groups need tool_handlers={"function": fn} (Python).
Needs: pip install 'evalkit[aws]'.
"""
from __future__ import annotations

import hashlib
import importlib
import time
import uuid
from typing import Any, Callable

from ...errors import ConfigError, PluginDependencyError
from ...plugins import HealthStatus, timed_health
from ...types import AgentResult, RetrievedDoc, ToolCall
from ..base import AgentAdapter


_PLUGIN = "system:bedrock_agent"
_MAX = 4000
# location type -> (key in the location object, field holding the document address)
_LOCATIONS = (("s3Location", "uri"), ("webLocation", "url"), ("confluenceLocation", "url"),
              ("salesforceLocation", "url"), ("sharePointLocation", "url"),
              ("kendraDocumentLocation", "uri"), ("customDocumentLocation", "id"))
_TRACE_PARTS = ("preProcessingTrace", "orchestrationTrace", "postProcessingTrace",
                "routingClassifierTrace", "customOrchestrationTrace")


def _boto3():
    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError as e:
        raise PluginDependencyError(_PLUGIN, e.name or "boto3", "aws") from e


def _reference_doc(ref: dict, cited: bool) -> RetrievedDoc:
    """A Bedrock retrievedReference -> RetrievedDoc (id = source URI when known)."""
    content = ref.get("content") or {}
    text = content.get("text") or ""
    loc = ref.get("location") or {}
    meta = dict(ref.get("metadata") or {})
    doc_id = None
    for key, attr in _LOCATIONS:
        if isinstance(loc.get(key), dict) and loc[key].get(attr):
            doc_id = loc[key][attr]
            break
    doc_id = (doc_id or meta.get("x-amz-bedrock-kb-source-uri") or meta.get("x-amz-bedrock-kb-chunk-id")
              or "sha1:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])
    score = meta.pop("score", None)
    meta.update(location_type=loc.get("type"), cited=cited, source="bedrock_kb")
    return RetrievedDoc(id=str(doc_id), text=text, score=score if isinstance(score, (int, float)) else None,
                        metadata=meta)


def _params(params: list | None) -> dict:
    return {p.get("name"): p.get("value") for p in (params or []) if isinstance(p, dict) and p.get("name")}


class _Turn:
    """Mutable state while reading one or more InvokeAgent streams."""

    def __init__(self):
        self.texts: list[str] = []
        self.trace: list[ToolCall] = []
        self.retrieved: list[RetrievedDoc] = []
        self.pending: dict[str, ToolCall] = {}
        self.prompt_tokens = self.completion_tokens = 0
        self.have_tokens = False
        self.final_response: str | None = None
        self.failure: str | None = None
        self.guardrail: str | None = None

    def add_doc(self, doc: RetrievedDoc) -> None:
        for d in self.retrieved:
            if d.id == doc.id:
                d.metadata["cited"] = d.metadata.get("cited") or doc.metadata.get("cited")
                if not d.text:
                    d.text = doc.text
                return
        self.retrieved.append(doc)


class BedrockAgent(AgentAdapter):
    def __init__(self, agent_id: str, agent_alias_id: str, region: str | None = None,
                 profile: str | None = None, session_attributes: dict | None = None,
                 prompt_session_attributes: dict | None = None, memory_id: str | None = None,
                 enable_trace: bool = True, timeout: float = 120, max_return_control_rounds: int = 5,
                 tool_handlers: dict[str, Callable] | None = None, client=None, control_client=None, **_):
        if not agent_id or not agent_alias_id:
            raise ConfigError("bedrock_agent needs 'agent_id' and 'agent_alias_id' (Bedrock console -> "
                              "Agents -> your agent; the draft alias is TSTALIASID)")
        self._session = None
        if client is None:
            boto3 = _boto3()
            from botocore.config import Config
            self._session = boto3.Session(profile_name=profile or None, region_name=region or None)
            client = self._session.client("bedrock-agent-runtime", region_name=region or None,
                                          config=Config(read_timeout=max(int(timeout), 1),
                                                        retries={"max_attempts": 4, "mode": "adaptive"}))
        self.client, self.control_client = client, control_client
        self.agent_id, self.agent_alias_id, self.region = agent_id, agent_alias_id, region
        self.session_attributes = dict(session_attributes or {})
        self.prompt_session_attributes = dict(prompt_session_attributes or {})
        self.memory_id, self.enable_trace, self.timeout = memory_id, enable_trace, float(timeout)
        self.max_return_control_rounds = max_return_control_rounds
        self.tool_handlers = dict(tool_handlers or {})

    # ---- InvokeAgent (bedrock-agent-runtime, API version 2023-07-26) ----
    def _invoke(self, session_id: str, text: str | None, state: _Turn, session_state: dict | None) -> dict | None:
        kw: dict[str, Any] = {"agentId": self.agent_id, "agentAliasId": self.agent_alias_id,
                              "sessionId": session_id, "enableTrace": self.enable_trace}
        if text is not None:
            kw["inputText"] = text
        ss = dict(session_state or {})
        if self.session_attributes:
            ss.setdefault("sessionAttributes", self.session_attributes)
        if self.prompt_session_attributes:
            ss.setdefault("promptSessionAttributes", self.prompt_session_attributes)
        if ss:
            kw["sessionState"] = ss
        if self.memory_id:
            kw["memoryId"] = self.memory_id
        resp = self.client.invoke_agent(**kw)
        deadline = time.monotonic() + self.timeout
        return_control = None
        for event in resp.get("completion") or []:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Bedrock agent {self.agent_id} session {session_id}: response not "
                                   f"complete after {self.timeout}s; raise 'timeout' if this is expected")
            if "chunk" in event:
                chunk = event["chunk"] or {}
                b = chunk.get("bytes") or b""
                state.texts.append(b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b))
                for cit in (chunk.get("attribution") or {}).get("citations") or []:
                    for ref in cit.get("retrievedReferences") or []:
                        state.add_doc(_reference_doc(ref, cited=True))
            elif "trace" in event:
                self._read_trace(event["trace"] or {}, state)
            elif "returnControl" in event:
                return_control = event["returnControl"] or {}
            elif "files" in event:
                continue
            else:
                for key, val in event.items():
                    if key.endswith("Exception"):
                        msg = val.get("message") if isinstance(val, dict) else val
                        raise RuntimeError(f"Bedrock agent stream error {key}: {msg}")
        return return_control

    def _read_trace(self, part: dict, state: _Turn) -> None:
        inner = part.get("trace") or {}
        collaborator = part.get("collaboratorName")
        if inner.get("failureTrace"):
            state.failure = (inner["failureTrace"] or {}).get("failureReason") or "unknown failure"
        g = inner.get("guardrailTrace")
        if g and g.get("action"):
            state.guardrail = g["action"]
        for name in _TRACE_PARTS:
            t = inner.get(name)
            if not isinstance(t, dict):
                continue
            mio = t.get("modelInvocationOutput") or {}
            usage = (mio.get("metadata") or {}).get("usage") or {}
            if usage.get("inputTokens") is not None or usage.get("outputTokens") is not None:
                state.have_tokens = True
                state.prompt_tokens += int(usage.get("inputTokens") or 0)
                state.completion_tokens += int(usage.get("outputTokens") or 0)
            if t.get("invocationInput"):
                self._invocation(t["invocationInput"], state, collaborator)
            if t.get("observation"):
                self._observation(t["observation"], state)

    @staticmethod
    def _invocation(inv: dict, state: _Turn, collaborator: str | None) -> None:
        itype = inv.get("invocationType")
        call: ToolCall | None = None
        if itype == "ACTION_GROUP" or inv.get("actionGroupInvocationInput"):
            ag = inv.get("actionGroupInvocationInput") or {}
            args = _params(ag.get("parameters"))
            for props in ((ag.get("requestBody") or {}).get("content") or {}).values():
                args.update(_params(props.get("properties") if isinstance(props, dict) else props))
            args["_action_group"] = ag.get("actionGroupName")
            if ag.get("apiPath"):
                args["_api"] = f"{(ag.get('verb') or '').upper()} {ag['apiPath']}".strip()
            call = ToolCall(str(ag.get("function") or ag.get("apiPath") or ag.get("actionGroupName") or "action"), args)
        elif itype == "KNOWLEDGE_BASE" or inv.get("knowledgeBaseLookupInput"):
            kb = inv.get("knowledgeBaseLookupInput") or {}
            call = ToolCall("knowledge_base", {"query": kb.get("text"), "knowledge_base_id": kb.get("knowledgeBaseId")})
        elif itype == "ACTION_GROUP_CODE_INTERPRETER" or inv.get("codeInterpreterInvocationInput"):
            ci = inv.get("codeInterpreterInvocationInput") or {}
            call = ToolCall("code_interpreter", {"code": ci.get("code"), "files": ci.get("files")})
        elif itype == "AGENT_COLLABORATOR" or inv.get("agentCollaboratorInvocationInput"):
            ac = inv.get("agentCollaboratorInvocationInput") or {}
            call = ToolCall(f"agent:{ac.get('agentCollaboratorName') or 'collaborator'}",
                            {"input": ((ac.get("input") or {}).get("text"))})
        if call is None:
            return
        if collaborator:
            call.args["_collaborator"] = collaborator
        state.trace.append(call)
        if inv.get("traceId"):
            state.pending[inv["traceId"]] = call

    @staticmethod
    def _observation(obs: dict, state: _Turn) -> None:
        call = state.pending.pop(obs.get("traceId"), None) if obs.get("traceId") else None
        if call is None:  # fall back to the most recent call without a result
            call = next((c for c in reversed(state.trace) if not c.result), None)
        result = None
        if obs.get("actionGroupInvocationOutput") is not None:
            result = (obs["actionGroupInvocationOutput"] or {}).get("text")
        elif obs.get("knowledgeBaseLookupOutput") is not None:
            refs = (obs["knowledgeBaseLookupOutput"] or {}).get("retrievedReferences") or []
            for ref in refs:
                state.add_doc(_reference_doc(ref, cited=False))
            result = f"{len(refs)} reference(s)"
        elif obs.get("codeInterpreterInvocationOutput") is not None:
            ci = obs["codeInterpreterInvocationOutput"] or {}
            result = ci.get("executionOutput") or ci.get("executionError")
        elif obs.get("agentCollaboratorInvocationOutput") is not None:
            result = ((obs["agentCollaboratorInvocationOutput"] or {}).get("output") or {}).get("text")
        elif obs.get("finalResponse") is not None:
            state.final_response = (obs["finalResponse"] or {}).get("text")
            return
        if call is not None and result is not None:
            call.result = str(result)[:_MAX]

    def _return_control_results(self, rc: dict, state: _Turn) -> tuple[list | None, str | None]:
        """Record return-of-control calls; run tool_handlers if all are available."""
        results, missing = [], []
        for item in rc.get("invocationInputs") or []:
            fi, ai = item.get("functionInvocationInput"), item.get("apiInvocationInput")
            src = fi or ai or {}
            name = (fi or {}).get("function") or (ai or {}).get("apiPath") or "action"
            args = _params(src.get("parameters"))
            for props in ((src.get("requestBody") or {}).get("content") or {}).values():
                args.update(_params(props.get("properties") if isinstance(props, dict) else props))
            args["_action_group"] = src.get("actionGroup")
            call = ToolCall(str(name), args)
            state.trace.append(call)
            if name not in self.tool_handlers:
                missing.append(name)
                continue
            out = self.tool_handlers[name]({k: v for k, v in args.items() if not k.startswith("_")})
            out = out if isinstance(out, str) else str(out)
            call.result = out[:_MAX]
            if fi:
                results.append({"functionResult": {"actionGroup": fi.get("actionGroup"), "function": name,
                                                   "responseBody": {"TEXT": {"body": out}}}})
            else:
                results.append({"apiResult": {"actionGroup": ai.get("actionGroup"), "apiPath": name,
                                              "httpMethod": ai.get("httpMethod"), "httpStatusCode": 200,
                                              "responseBody": {"application/json": {"body": out}}}})
        if missing:
            return None, (f"agent returned control for {missing} (return-of-control action group) but no "
                          f"tool_handlers were given for them")
        return results, None

    def _turn(self, session_id: str, text: str, state: _Turn) -> str | None:
        """One user turn, including return-of-control rounds. Returns an error or None."""
        rc = self._invoke(session_id, text, state, None)
        rounds = 0
        while rc:
            if rounds >= self.max_return_control_rounds:
                return f"stopped after {rounds} return-of-control rounds"
            results, err = self._return_control_results(rc, state)
            if err:
                return err
            rounds += 1
            rc = self._invoke(session_id, None, state, {"invocationId": rc.get("invocationId"),
                                                        "returnControlInvocationResults": results})
        return None

    def run(self, input: str) -> AgentResult:
        return self.run_conversation([input])

    def run_conversation(self, turns: list[str]) -> AgentResult:
        t0 = time.monotonic()
        session_id = f"evalkit-{uuid.uuid4().hex}"
        trace: list[ToolCall] = []
        retrieved: list[RetrievedDoc] = []
        pt = ct = 0
        have_tokens = False
        answer, error, raw = "", None, {"session_id": session_id, "agent_id": self.agent_id,
                                        "agent_alias_id": self.agent_alias_id}
        for turn in turns:
            state = _Turn()
            error = self._turn(session_id, turn, state)
            answer = "".join(state.texts).strip() or (state.final_response or "").strip()
            trace += state.trace
            for d in state.retrieved:
                if all(d.id != r.id for r in retrieved):
                    retrieved.append(d)
            if state.have_tokens:
                have_tokens = True
                pt += state.prompt_tokens
                ct += state.completion_tokens
            if state.guardrail:
                raw["guardrail_action"] = state.guardrail
            if state.failure and not error:
                raw["failure_reason"] = state.failure
                if not answer:
                    error = f"Bedrock agent failure: {state.failure}"
            if error:
                break
        if len(turns) > 1:
            raw["turns"] = len(turns)
        return AgentResult(answer=answer, trace=trace, retrieved=retrieved,
                           prompt_tokens=pt if have_tokens else None,
                           completion_tokens=ct if have_tokens else None,
                           latency_s=round(time.monotonic() - t0, 3), error=error, raw=raw)

    def health_check(self) -> HealthStatus:
        control = self.control_client
        if control is None and self._session is not None:
            control = self._session.client("bedrock-agent", region_name=self.region or None)
        if control is None:
            return super().health_check()

        def probe():
            alias = control.get_agent_alias(agentId=self.agent_id, agentAliasId=self.agent_alias_id)
            a = alias.get("agentAlias") or {}
            status = a.get("agentAliasStatus", "?")
            if status not in ("PREPARED", "?"):
                raise RuntimeError(f"agent alias {self.agent_alias_id} status is {status}")
            return f"agent alias '{a.get('agentAliasName', self.agent_alias_id)}' {status}"
        return timed_health(probe)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
