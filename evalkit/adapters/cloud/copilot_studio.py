"""Microsoft Copilot Studio agents through the Bot Framework Direct Line 3.0 REST API.

config (Direct Line secret: Copilot Studio -> Settings -> Security -> Web channel security):
  type: copilot_studio
  direct_line_secret: ${secret:copilot-directline-secret}
  domain: https://directline.botframework.com/v3/directline   # regional, e.g. europe.directline.botframework.com
  user_id: evalkit-user
  locale: en-US
  timeout: 60               # max seconds to wait for the agent's reply
  poll_interval: 1.0
  settle_s: 3.0             # after the first reply, wait this long for follow-up messages
  send_start_event: false   # send a startConversation event first (triggers the Greeting topic)

config (token endpoint, no secret: Copilot Studio -> Channels -> Mobile app):
  type: copilot_studio
  token_endpoint: https://<env>.environment.api.powerplatform.com/powervirtualagents/botsbyschema/<schema>/directline/token?api-version=2022-03-01-preview

Flow: token (POST /tokens/generate with the secret, or GET token_endpoint) ->
POST /conversations -> POST /conversations/{id}/activities -> poll
GET /conversations/{id}/activities?watermark=... until the bot replies.
Each test case is a new conversation; a multi-turn case reuses one.
"""
from __future__ import annotations

import time
from typing import Any

from ...errors import ConfigError
from ...plugins import HealthStatus, timed_health
from ...types import AgentResult, RetrievedDoc, ToolCall
from ..base import AgentAdapter

_DEFAULT_DOMAIN = "https://directline.botframework.com/v3/directline"
_MAX = 4000


def _card_text(card: Any) -> list[str]:
    """Collect TextBlock / FactSet text from an Adaptive Card body (recursive)."""
    out: list[str] = []
    if isinstance(card, dict):
        if card.get("type") == "TextBlock" and card.get("text"):
            out.append(str(card["text"]))
        if card.get("type") == "FactSet":
            out += [f"{f.get('title', '')}: {f.get('value', '')}" for f in card.get("facts") or []]
        for key in ("body", "items", "columns", "actions", "card"):
            out += _card_text(card.get(key))
    elif isinstance(card, list):
        for c in card:
            out += _card_text(c)
    return out


class CopilotStudioAgent(AgentAdapter):
    def __init__(self, direct_line_secret: str | None = None, token_endpoint: str | None = None,
                 token: str | None = None, domain: str = _DEFAULT_DOMAIN, user_id: str = "evalkit-user",
                 user_name: str | None = None, locale: str = "en-US", timeout: float = 60,
                 poll_interval: float = 1.0, settle_s: float = 3.0, send_start_event: bool = False,
                 include_card_text: bool = True, session=None, **_):
        if not (direct_line_secret or token_endpoint or token):
            raise ConfigError("copilot_studio needs 'direct_line_secret' (Copilot Studio -> Settings -> Security "
                              "-> Web channel security) or 'token_endpoint' (Channels -> Mobile app)")
        self.secret, self.token_endpoint, self.static_token = direct_line_secret, token_endpoint, token
        self.domain = domain.rstrip("/")
        self.user_id, self.user_name, self.locale = user_id, user_name, locale
        self.timeout, self.poll_interval, self.settle_s = float(timeout), float(poll_interval), float(settle_s)
        self.send_start_event, self.include_card_text = send_start_event, include_card_text
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    # ---- HTTP helpers ----
    def _req(self, method: str, url: str, bearer: str | None, body: dict | None = None, ok=(200, 201)):
        headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
        r = self.session.request(method, url, json=body, headers=headers, timeout=min(self.timeout, 30) or 30)
        if r.status_code not in ok:
            hint = " (check the Direct Line secret / token)" if r.status_code in (401, 403) else ""
            raise RuntimeError(f"Direct Line {method} {url.split('?')[0]} failed: HTTP {r.status_code}{hint} "
                               f"{r.text[:300]}")
        return r.json() if r.text else {}

    def _get_token(self) -> str:
        if self.static_token:
            return self.static_token
        if self.token_endpoint:
            data = self._req("GET", self.token_endpoint, None)
        else:
            data = self._req("POST", f"{self.domain}/tokens/generate", self.secret,
                             {"user": {"id": self.user_id, **({"name": self.user_name} if self.user_name else {})}})
        tok = data.get("token")
        if not tok:
            raise RuntimeError(f"Direct Line token response had no 'token': {str(data)[:200]}")
        return tok

    def _start(self) -> tuple[str, str]:
        token = self._get_token()
        conv = self._req("POST", f"{self.domain}/conversations", token)
        cid = conv.get("conversationId")
        if not cid:
            raise RuntimeError(f"Direct Line start conversation returned no conversationId: {str(conv)[:200]}")
        return cid, conv.get("token") or token

    def _post(self, cid: str, token: str, activity: dict) -> str:
        activity = {"from": {"id": self.user_id, **({"name": self.user_name} if self.user_name else {})},
                    "locale": self.locale, **activity}
        return self._req("POST", f"{self.domain}/conversations/{cid}/activities", token, activity).get("id", "")

    def _poll(self, cid: str, token: str, watermark: str | None, reply_to: str | None):
        """Poll until bot replies arrive and settle, or the timeout passes.
        Returns (bot activities, new watermark)."""
        deadline = time.monotonic() + self.timeout
        replies: list[dict] = []
        last_new = None
        ended = False
        while True:
            url = f"{self.domain}/conversations/{cid}/activities" + (f"?watermark={watermark}" if watermark else "")
            data = self._req("GET", url, token)
            watermark = data.get("watermark") or watermark
            for act in data.get("activities") or []:
                if (act.get("from") or {}).get("id") == self.user_id:
                    continue
                if reply_to and act.get("replyToId") not in (None, reply_to):
                    continue
                replies.append(act)
                last_new = time.monotonic()
                if act.get("type") == "endOfConversation":
                    ended = True
            has_message = any(a.get("type") == "message" for a in replies)
            now = time.monotonic()
            if ended or (has_message and last_new is not None and now - last_new >= self.settle_s):
                return replies, watermark
            if now >= deadline:
                if has_message:
                    return replies, watermark
                raise TimeoutError(f"Copilot Studio did not reply within {self.timeout}s "
                                   f"(conversation {cid}); raise 'timeout' if the agent is slow")
            if self.poll_interval:
                time.sleep(self.poll_interval)

    # ---- parsing ----
    def _parse(self, replies: list[dict]) -> tuple[str, list[ToolCall], list[RetrievedDoc]]:
        texts: list[str] = []
        trace: list[ToolCall] = []
        docs: list[RetrievedDoc] = []
        for act in replies:
            typ = act.get("type")
            if typ == "message":
                if act.get("text"):
                    texts.append(str(act["text"]))
                if self.include_card_text:
                    for att in act.get("attachments") or []:
                        if "adaptive" in str(att.get("contentType", "")):
                            texts += _card_text(att.get("content"))
                for ent in act.get("entities") or []:
                    for cit in ent.get("citation") or []:          # schema.org Claim citations
                        app = cit.get("appearance") or {}
                        doc_id = app.get("url") or app.get("name") or f"citation-{cit.get('position')}"
                        if all(d.id != doc_id for d in docs):
                            docs.append(RetrievedDoc(id=str(doc_id),
                                                     text=str(app.get("abstract") or app.get("text") or ""),
                                                     metadata={"title": app.get("name"), "cited": True,
                                                               "position": cit.get("position")}))
            elif typ == "event":
                # Generative orchestration emits plan/step events (names vary by release).
                name = str(act.get("name") or act.get("valueType") or "")
                value = act.get("value") if isinstance(act.get("value"), dict) else {}
                if "PlanStepTriggered" in name or "PlanStepFinished" in name:
                    tool = value.get("taskDialogId") or value.get("stepId") or name
                    if "Finished" in name:
                        prev = next((t for t in reversed(trace) if t.tool == tool and not t.result), None)
                        if prev is not None:
                            prev.result = str(value.get("observation") or value.get("state") or "finished")[:_MAX]
                            continue
                    trace.append(ToolCall(str(tool), dict(value.get("arguments") or {}),
                                          str(value.get("observation") or "")[:_MAX]))
        return "\n".join(t for t in texts if t).strip(), trace, docs

    # ---- AgentAdapter ----
    def run(self, input: str) -> AgentResult:
        return self.run_conversation([input])

    def run_conversation(self, turns: list[str]) -> AgentResult:
        t0 = time.monotonic()
        cid, token = self._start()
        watermark = None
        if self.send_start_event:
            eid = self._post(cid, token, {"type": "event", "name": "startConversation"})
            _, watermark = self._poll(cid, token, None, eid)   # discard the greeting
        trace: list[ToolCall] = []
        retrieved: list[RetrievedDoc] = []
        answer = ""
        n_activities = 0
        for turn in turns:
            aid = self._post(cid, token, {"type": "message", "text": turn})
            replies, watermark = self._poll(cid, token, watermark, aid)
            n_activities += len(replies)
            answer, t, docs = self._parse(replies)
            trace += t
            for d in docs:
                if all(d.id != r.id for r in retrieved):
                    retrieved.append(d)
        raw = {"conversation_id": cid, "bot_activities": n_activities}
        if len(turns) > 1:
            raw["turns"] = len(turns)
        return AgentResult(answer=answer, trace=trace, retrieved=retrieved,
                           latency_s=round(time.monotonic() - t0, 3), raw=raw)

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"Direct Line token issued ({len(self._get_token())} chars)")

    def close(self) -> None:
        self.session.close()
