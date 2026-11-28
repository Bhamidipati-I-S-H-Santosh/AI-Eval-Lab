"""Cloud agent-system adapters, tested against fake SDK clients / HTTP sessions
shaped like each vendor's documented responses. No network, no credentials."""
from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace as NS

import pytest

from evalkit.errors import ConfigError, PluginDependencyError
from evalkit.plugins import REGISTRY


# ====================================================================== OpenAI Responses

class FakeResponses:
    def __init__(self, responses):
        self.queue, self.calls = list(responses), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.queue.pop(0)


def _resp(rid, output, status="completed", usage=(10, 5), **extra):
    return NS(id=rid, status=status, output=output, output_text=None,
              usage=NS(input_tokens=usage[0], output_tokens=usage[1]), **extra)


def _msg(text, annotations=()):
    return NS(type="message", role="assistant",
              content=[NS(type="output_text", text=text, annotations=list(annotations))])


def test_openai_responses_answer_tools_citations_usage():
    resp = _resp("resp_1", [
        NS(type="file_search_call", id="fs_1", queries=["refund policy"], status="completed",
           results=[NS(file_id="file-a", filename="refunds.md", score=0.91, text="Refunds within 30 days.",
                       attributes={})]),
        NS(type="web_search_call", id="ws_1", status="completed", action=NS(type="search", query="refund law")),
        _msg("Refunds are accepted within 30 days.",
             [NS(type="file_citation", file_id="file-a", filename="refunds.md", index=10),
              NS(type="url_citation", url="https://example.com/law", title="Law", start_index=0, end_index=5)]),
    ])
    fake = NS(responses=FakeResponses([resp]))
    agent = REGISTRY.build("system", {"type": "openai_responses", "model": "gpt-4.1",
                                      "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
                                      "client": fake})
    r = agent.run("What is the refund window?")
    assert r.answer == "Refunds are accepted within 30 days."
    assert r.tools_used == ["file_search", "web_search"]
    assert r.trace[0].args == {"queries": ["refund policy"]}
    ids = [d.id for d in r.retrieved]
    assert ids == ["file-a", "https://example.com/law"]
    assert r.retrieved[0].text == "Refunds within 30 days." and r.retrieved[0].metadata["cited"] is True
    assert (r.prompt_tokens, r.completion_tokens) == (10, 5)
    assert r.raw["response_ids"] == ["resp_1"]
    call = fake.responses.calls[0]
    assert call["include"] == ["file_search_call.results"] and call["input"] == "What is the refund window?"
    assert "previous_response_id" not in call


def test_openai_responses_multi_turn_chains_previous_response_id():
    fake = NS(responses=FakeResponses([_resp("resp_1", [_msg("Hi Ana.")]),
                                       _resp("resp_2", [_msg("Your name is Ana.")])]))
    agent = REGISTRY.build("system", {"type": "openai_responses", "model": "gpt-4.1", "client": fake})
    r = agent.run_conversation(["My name is Ana.", "What is my name?"])
    assert r.answer == "Your name is Ana."
    assert fake.responses.calls[1]["previous_response_id"] == "resp_1"
    assert (r.prompt_tokens, r.completion_tokens) == (20, 10)


def test_openai_responses_function_tools_and_failure():
    fc = NS(type="function_call", name="get_order", arguments='{"order_id": "42"}', call_id="call_1")
    fake = NS(responses=FakeResponses([_resp("resp_1", [fc]), _resp("resp_2", [_msg("Order 42 shipped.")])]))
    agent = REGISTRY.build("system", {"type": "openai_responses", "model": "m", "client": fake,
                                      "tool_handlers": {"get_order": lambda a: f"order {a['order_id']}: shipped"}})
    r = agent.run("Where is order 42?")
    assert r.answer == "Order 42 shipped." and r.error is None
    assert r.trace[0].tool == "get_order" and r.trace[0].args == {"order_id": "42"}
    assert r.trace[0].result == "order 42: shipped"
    second = fake.responses.calls[1]
    assert second["previous_response_id"] == "resp_1"
    assert second["input"] == [{"type": "function_call_output", "call_id": "call_1", "output": "order 42: shipped"}]

    # no handler -> recorded in the trace, reported as an error
    fake2 = NS(responses=FakeResponses([_resp("resp_1", [fc])]))
    r2 = REGISTRY.build("system", {"type": "openai_responses", "model": "m", "client": fake2}).run("x")
    assert r2.tools_used == ["get_order"] and "tool_handlers" in r2.error

    failed = _resp("resp_f", [], status="failed", error=NS(code="server_error", message="boom"))
    r3 = REGISTRY.build("system", {"type": "openai_responses", "model": "m",
                                   "client": NS(responses=FakeResponses([failed]))}).run("x")
    assert "failed" in r3.error and "boom" in r3.error


def test_openai_responses_requires_model():
    with pytest.raises(ConfigError):
        REGISTRY.build("system", {"type": "openai_responses", "client": NS()})


# ====================================================================== thread/run fakes

class RunBackend:
    """In-memory Assistants-style service shared by the OpenAI and Azure fakes."""

    def __init__(self, statuses=("in_progress", "completed"), reply="Done.", annotations=(),
                 steps=(), usage=(30, 7), last_error=None, required_action=None):
        self.statuses, self.reply, self.annotations = list(statuses), reply, list(annotations)
        self.steps, self.usage, self.last_error = list(steps), usage, last_error
        self.required_action = required_action
        self.threads, self.messages, self.runs = [], {}, {}
        self.cancelled, self.deleted, self.submitted, self.created_runs = [], [], [], []
        self._polls = {}

    def create_thread(self):
        tid = f"thread_{len(self.threads) + 1}"
        self.threads.append(tid)
        self.messages[tid] = []
        return NS(id=tid)

    def post(self, thread_id, role, content):
        self.messages[thread_id].append(NS(id=f"msg_{len(self.messages[thread_id])}", role=role, run_id=None,
                                           content=[NS(type="text", text=NS(value=content, annotations=[]))]))
        return self.messages[thread_id][-1]

    def _run(self, thread_id, run_id):
        status = self.runs[run_id]["statuses"][min(self._polls[run_id], len(self.runs[run_id]["statuses"]) - 1)]
        done = status in ("completed", "failed", "cancelled", "expired")
        ra = self.required_action if status == "requires_action" else None
        return NS(id=run_id, thread_id=thread_id, status=status, required_action=ra,
                  usage=NS(prompt_tokens=self.usage[0], completion_tokens=self.usage[1]) if done else None,
                  last_error=self.last_error if status == "failed" else None)

    def create_run(self, thread_id, **kw):
        run_id = f"run_{len(self.runs) + 1}"
        self.created_runs.append({"thread_id": thread_id, **kw})
        self.runs[run_id] = {"statuses": list(self.statuses), "thread": thread_id}
        self._polls[run_id] = 0
        n = sum(1 for m in self.messages[thread_id] if m.role == "assistant") + 1
        self.messages[thread_id].append(NS(
            id=f"msg_a{n}", role="assistant", run_id=run_id,
            content=[NS(type="text", text=NS(value=f"{self.reply}" if n == 1 else f"{self.reply} (turn {n})",
                                             annotations=list(self.annotations)))]))
        return self._run(thread_id, run_id)

    def get_run(self, thread_id, run_id):
        self._polls[run_id] += 1
        return self._run(thread_id, run_id)

    def cancel(self, thread_id, run_id):
        self.cancelled.append(run_id)
        self.runs[run_id]["statuses"] = ["cancelled"]
        return self._run(thread_id, run_id)

    def submit(self, thread_id, run_id, **kw):
        self.submitted.append(kw)
        self._polls[run_id] += 1
        return self._run(thread_id, run_id)

    def list_messages(self, thread_id, run_id=None):
        return [m for m in self.messages[thread_id] if run_id is None or m.run_id == run_id]

    def list_steps(self):
        return list(self.steps)


def _fake_openai(b: RunBackend):
    steps_api = NS(list=lambda run_id, thread_id, **kw: NS(data=b.list_steps()))
    runs = NS(create=lambda thread_id, **kw: b.create_run(thread_id, **kw),
              retrieve=lambda run_id, thread_id: b.get_run(thread_id, run_id),
              cancel=lambda run_id, thread_id: b.cancel(thread_id, run_id),
              submit_tool_outputs=lambda run_id, thread_id, **kw: b.submit(thread_id, run_id, **kw),
              steps=steps_api)
    messages = NS(create=lambda thread_id, role, content: b.post(thread_id, role, content),
                  list=lambda thread_id, run_id=None, order="desc": NS(data=b.list_messages(thread_id, run_id)))
    threads = NS(create=b.create_thread, delete=lambda thread_id: b.deleted.append(thread_id),
                 messages=messages, runs=runs)
    assistants = NS(retrieve=lambda assistant_id: NS(id=assistant_id, name="Support", model="gpt-4o"))
    return NS(beta=NS(threads=threads, assistants=assistants))


def _tool_step(*calls):
    return NS(type="tool_calls", step_details=NS(type="tool_calls", tool_calls=list(calls)))


# ====================================================================== OpenAI Assistants

def test_openai_assistants_run_steps_citations_usage_cleanup():
    fs = NS(id="call_fs", type="file_search", file_search=NS(results=[
        NS(file_id="file-1", file_name="policy.pdf", score=0.8, content=[NS(type="text", text="30-day refunds")])]))
    fn = NS(id="call_fn", type="function", function=NS(name="lookup", arguments='{"q": "x"}', output="42"))
    ann = NS(type="file_citation", text="【4:0†policy.pdf】", file_citation=NS(file_id="file-1"),
             start_index=20, end_index=36)
    b = RunBackend(reply="Refunds take 30 days【4:0†policy.pdf】.", annotations=[ann], steps=[_tool_step(fs, fn)])
    agent = REGISTRY.build("system", {"type": "openai_assistants", "assistant_id": "asst_1",
                                      "poll_interval": 0, "client": _fake_openai(b)})
    r = agent.run("How long do refunds take?")
    assert r.error is None
    assert r.answer == "Refunds take 30 days."
    assert r.tools_used == ["file_search", "lookup"]
    assert r.trace[1].args == {"q": "x"} and r.trace[1].result == "42"
    assert [d.id for d in r.retrieved] == ["file-1"]
    assert r.retrieved[0].text == "30-day refunds" and r.retrieved[0].metadata["cited"] is True
    assert (r.prompt_tokens, r.completion_tokens) == (30, 7)
    assert r.raw["thread_id"] == "thread_1" and r.raw["run_ids"] == ["run_1"]
    assert b.deleted == ["thread_1"]
    assert agent.health_check().ok


def test_openai_assistants_multi_turn_reuses_thread():
    b = RunBackend(statuses=["completed"])
    agent = REGISTRY.build("system", {"type": "openai_assistants", "assistant_id": "asst_1",
                                      "poll_interval": 0, "client": _fake_openai(b)})
    r = agent.run_conversation(["My name is Ana.", "What is my name?"])
    assert b.threads == ["thread_1"]
    assert {c["thread_id"] for c in b.created_runs} == {"thread_1"} and len(b.created_runs) == 2
    assert r.answer == "Done. (turn 2)"
    assert (r.prompt_tokens, r.completion_tokens) == (60, 14)


def test_openai_assistants_failed_run_and_timeout():
    b = RunBackend(statuses=["in_progress", "failed"], last_error=NS(code="rate_limit_exceeded", message="slow down"))
    r = REGISTRY.build("system", {"type": "openai_assistants", "assistant_id": "a", "poll_interval": 0,
                                  "client": _fake_openai(b)}).run("x")
    assert "failed" in r.error and "rate_limit_exceeded" in r.error

    stuck = RunBackend(statuses=["in_progress"])
    agent = REGISTRY.build("system", {"type": "openai_assistants", "assistant_id": "a", "poll_interval": 0.01,
                                      "timeout": 0.05, "client": _fake_openai(stuck)})
    with pytest.raises(TimeoutError, match="did not finish"):
        agent.run("x")
    assert stuck.cancelled == ["run_1"] and stuck.deleted == ["thread_1"]


def test_openai_assistants_requires_action_with_and_without_handler():
    ra = NS(type="submit_tool_outputs", submit_tool_outputs=NS(tool_calls=[
        NS(id="call_1", type="function", function=NS(name="get_time", arguments="{}"))]))
    b = RunBackend(statuses=["requires_action", "completed"], required_action=ra)
    agent = REGISTRY.build("system", {"type": "openai_assistants", "assistant_id": "a", "poll_interval": 0,
                                      "client": _fake_openai(b), "tool_handlers": {"get_time": lambda a: "noon"}})
    assert agent.run("time?").error is None
    assert b.submitted == [{"tool_outputs": [{"tool_call_id": "call_1", "output": "noon"}]}]

    b2 = RunBackend(statuses=["requires_action"], required_action=ra)
    r = REGISTRY.build("system", {"type": "openai_assistants", "assistant_id": "a", "poll_interval": 0,
                                  "client": _fake_openai(b2)}).run("time?")
    assert "get_time" in r.error and b2.cancelled == ["run_1"]


def test_openai_missing_sdk_raises_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)       # simulate "not installed"
    with pytest.raises(PluginDependencyError, match="evalkit\\[openai\\]"):
        REGISTRY.build("system", {"type": "openai_assistants", "assistant_id": "asst_1"})
    with pytest.raises(PluginDependencyError):
        REGISTRY.build("system", {"type": "openai_responses", "model": "gpt-4.1"})


# ====================================================================== Azure AI Foundry

def _fake_azure(b: RunBackend, with_steps_include=True):
    def list_steps(thread_id, run_id, **kw):
        if kw and not with_steps_include:
            raise TypeError("unexpected keyword 'include'")
        return iter(b.list_steps())                      # ItemPaged-like iterable
    agents = NS(
        threads=NS(create=b.create_thread, delete=lambda thread_id: b.deleted.append(thread_id)),
        messages=NS(create=lambda thread_id, role, content: b.post(thread_id, role, content),
                    list=lambda thread_id, order=None: iter(b.list_messages(thread_id))),
        runs=NS(create=lambda thread_id, agent_id, **kw: b.create_run(thread_id, agent_id=agent_id, **kw),
                get=lambda thread_id, run_id: b.get_run(thread_id, run_id),
                cancel=lambda thread_id, run_id: b.cancel(thread_id, run_id),
                submit_tool_outputs=lambda thread_id, run_id, **kw: b.submit(thread_id, run_id, **kw),
                create_and_process=lambda thread_id, agent_id, **kw: _process(b, thread_id, agent_id)),
        run_steps=NS(list=list_steps),
        get_agent=lambda agent_id: NS(id=agent_id, name="Contoso agent", model="gpt-4o"),
    )
    return NS(agents=agents)          # shaped like AIProjectClient


def _process(b, thread_id, agent_id):
    run = b.create_run(thread_id, agent_id=agent_id)
    while run.status not in ("completed", "failed", "cancelled", "expired", "requires_action"):
        run = b.get_run(thread_id, run.id)
    return run


AZ = {"type": "azure_foundry", "endpoint": "https://res.services.ai.azure.com/api/projects/p",
      "agent_id": "asst_az", "poll_interval": 0}


def test_azure_foundry_tools_citations_usage_via_registry():
    steps = [_tool_step(
        NS(id="c1", type="azure_ai_search", azure_ai_search={"input": "vacation policy", "output": "3 docs"}),
        NS(id="c2", type="bing_grounding",
           bing_grounding={"requesturl": "https://api.bing.microsoft.com/v7.0/search?q=pto+law"}),
        NS(id="c3", type="file_search", file_search=NS(results=[
            NS(file_id="assistant-f1", file_name="handbook.pdf", score=0.7,
               content=[NS(type="text", text="20 days PTO")])])),
        NS(id="c4", type="function", function=NS(name="get_balance", arguments='{"user": "ana"}', output="12")),
    ), NS(type="message_creation", step_details=NS(type="message_creation"))]
    anns = [NS(type="url_citation", text="【3:0†source】",
               url_citation=NS(url="https://contoso.sharepoint.com/hr/pto.docx", title="PTO policy")),
            NS(type="file_citation", text="【3:1†source】", file_citation=NS(file_id="assistant-f1", quote=None))]
    b = RunBackend(reply="You get 20 days【3:0†source】【3:1†source】.", annotations=anns, steps=steps)
    agent = REGISTRY.build("system", {**AZ, "client": _fake_azure(b)})
    r = agent.run("How much PTO do I get?")
    assert r.error is None
    assert r.answer == "You get 20 days."
    assert r.tools_used == ["azure_ai_search", "bing_grounding", "file_search", "get_balance"]
    assert r.trace[0].args == {"query": "vacation policy"} and r.trace[0].result == "3 docs"
    assert r.trace[1].args["query"] == "pto law"
    assert r.trace[3].args == {"user": "ana"} and r.trace[3].result == "12"
    assert [d.id for d in r.retrieved] == ["assistant-f1", "https://contoso.sharepoint.com/hr/pto.docx"]
    assert r.retrieved[0].text == "20 days PTO" and r.retrieved[0].metadata["cited"] is True
    assert (r.prompt_tokens, r.completion_tokens) == (30, 7)
    assert r.raw["thread_id"] == "thread_1" and b.created_runs[0]["agent_id"] == "asst_az"
    assert b.deleted == ["thread_1"]
    h = agent.health_check()
    assert h.ok and "Contoso agent" in h.detail


def test_azure_foundry_multi_turn_and_create_and_process():
    b = RunBackend(statuses=["queued", "in_progress", "completed"])
    agent = REGISTRY.build("system", {**AZ, "client": _fake_azure(b, with_steps_include=False),
                                      "use_create_and_process": True, "delete_threads": False})
    r = agent.run_conversation(["Hi, I'm Ana.", "Who am I?"])
    assert b.threads == ["thread_1"] and len(b.created_runs) == 2
    assert r.answer == "Done. (turn 2)" and r.raw["run_ids"] == ["run_1", "run_2"]
    assert (r.prompt_tokens, r.completion_tokens) == (60, 14)
    assert b.deleted == []


def test_azure_foundry_failed_run_timeout_and_approval():
    b = RunBackend(statuses=["failed"], last_error={"code": "server_error", "message": "tool crashed"})
    r = REGISTRY.build("system", {**AZ, "client": _fake_azure(b)}).run("x")
    assert "status 'failed'" in r.error and "tool crashed" in r.error

    stuck = RunBackend(statuses=["in_progress"])
    with pytest.raises(TimeoutError):
        REGISTRY.build("system", {**AZ, "client": _fake_azure(stuck), "timeout": 0.03,
                                  "poll_interval": 0.01}).run("x")
    assert stuck.cancelled == ["run_1"]

    ra = NS(type="submit_tool_approval", submit_tool_approval=NS(tool_calls=[NS(id="mcp_1", name="delete_repo")]))
    b2 = RunBackend(statuses=["requires_action"], required_action=ra)
    r2 = REGISTRY.build("system", {**AZ, "client": _fake_azure(b2)}).run("x")
    assert "approval" in r2.error and b2.cancelled == ["run_1"]
    b3 = RunBackend(statuses=["requires_action", "completed"], required_action=ra)
    r3 = REGISTRY.build("system", {**AZ, "client": _fake_azure(b3), "auto_approve_tools": True}).run("x")
    assert r3.error is None and b3.submitted == [{"tool_approvals": [{"tool_call_id": "mcp_1", "approve": True}]}]


def test_azure_foundry_preview_sdk_method_names():
    b = RunBackend(statuses=["completed"])
    flat = NS(create_thread=b.create_thread,
              create_message=lambda thread_id, role, content: b.post(thread_id, role, content),
              create_run=lambda thread_id, agent_id, **kw: b.create_run(thread_id, agent_id=agent_id),
              get_run=lambda thread_id, run_id: b.get_run(thread_id, run_id),
              list_messages=lambda thread_id, order=None: NS(data=b.list_messages(thread_id)),
              list_run_steps=lambda thread_id, run_id, **kw: NS(data=[]),
              delete_thread=lambda thread_id: b.deleted.append(thread_id))
    r = REGISTRY.build("system", {**AZ, "client": NS(agents=flat)}).run("x")
    assert r.answer == "Done." and b.deleted == ["thread_1"]


def test_azure_foundry_without_sdk_raises_dependency_error():
    if "azure.ai.projects" in sys.modules:
        pytest.skip("azure-ai-projects installed")
    with pytest.raises(PluginDependencyError, match="evalkit\\[azure\\]"):
        REGISTRY.build("system", {k: v for k, v in AZ.items()})
    with pytest.raises(ConfigError, match="agent_id"):
        REGISTRY.build("system", {"type": "azure_foundry", "endpoint": "https://x", "agent_id": "", "client": NS()})


# ====================================================================== AWS Bedrock Agents

class FakeBedrock:
    def __init__(self, *streams):
        self.streams, self.calls = list(streams), []

    def invoke_agent(self, **kw):
        self.calls.append(kw)
        return {"completion": self.streams.pop(0), "sessionId": kw["sessionId"], "contentType": "application/json"}


def _usage(i, o):
    return {"modelInvocationOutput": {"metadata": {"usage": {"inputTokens": i, "outputTokens": o}}}}


def _bedrock_stream():
    ref = {"content": {"text": "Returns accepted within 30 days."},
           "location": {"type": "S3", "s3Location": {"uri": "s3://kb/returns.pdf"}},
           "metadata": {"x-amz-bedrock-kb-chunk-id": "c1"}}
    other = {"content": {"text": "Shipping is free over $50."},
             "location": {"type": "WEB", "webLocation": {"url": "https://shop.example.com/shipping"}}}
    return [
        {"trace": {"trace": {"preProcessingTrace": _usage(50, 5)}}},
        {"trace": {"trace": {"orchestrationTrace": _usage(200, 30)}}},
        {"trace": {"trace": {"orchestrationTrace": {"rationale": {"text": "look up the order", "traceId": "t-0"}}}}},
        {"trace": {"trace": {"orchestrationTrace": {"invocationInput": {
            "invocationType": "ACTION_GROUP", "traceId": "t-0",
            "actionGroupInvocationInput": {"actionGroupName": "orders", "function": "get_order",
                                           "parameters": [{"name": "order_id", "type": "string", "value": "42"}]}}}}}},
        {"trace": {"trace": {"orchestrationTrace": {"observation": {
            "type": "ACTION_GROUP", "traceId": "t-0", "actionGroupInvocationOutput": {"text": "{\"status\": \"shipped\"}"}}}}}},
        {"trace": {"trace": {"orchestrationTrace": {"invocationInput": {
            "invocationType": "KNOWLEDGE_BASE", "traceId": "t-1",
            "knowledgeBaseLookupInput": {"text": "return policy", "knowledgeBaseId": "KB123"}}}}}},
        {"trace": {"trace": {"orchestrationTrace": {"observation": {
            "type": "KNOWLEDGE_BASE", "traceId": "t-1",
            "knowledgeBaseLookupOutput": {"retrievedReferences": [ref, other]}}}}}},
        {"trace": {"trace": {"orchestrationTrace": _usage(300, 40)}}},
        {"chunk": {"bytes": "Order 42 has shipped. ".encode(),
                   "attribution": {"citations": [{"generatedResponsePart": {"textResponsePart": {"text": "x"}},
                                                  "retrievedReferences": [ref]}]}}},
        {"chunk": {"bytes": "Returns are accepted within 30 days.".encode()}},
        {"trace": {"trace": {"orchestrationTrace": {"observation": {
            "type": "FINISH", "traceId": "t-2", "finalResponse": {"text": "Order 42 has shipped."}}}}}},
    ]


BR = {"type": "bedrock_agent", "agent_id": "AGENT1", "agent_alias_id": "TSTALIASID"}


def test_bedrock_answer_trace_citations_usage():
    fake = FakeBedrock(_bedrock_stream())
    agent = REGISTRY.build("system", {**BR, "client": fake, "session_attributes": {"tenant": "nw"}})
    r = agent.run("Where is order 42 and can I return it?")
    assert r.error is None
    assert r.answer == "Order 42 has shipped. Returns are accepted within 30 days."
    assert r.tools_used == ["get_order", "knowledge_base"]
    assert r.trace[0].args == {"order_id": "42", "_action_group": "orders"}
    assert r.trace[0].result == '{"status": "shipped"}'
    assert r.trace[1].args == {"query": "return policy", "knowledge_base_id": "KB123"}
    assert r.trace[1].result == "2 reference(s)"
    assert [d.id for d in r.retrieved] == ["s3://kb/returns.pdf", "https://shop.example.com/shipping"]
    assert r.retrieved[0].metadata["cited"] is True and r.retrieved[1].metadata["cited"] is False
    assert (r.prompt_tokens, r.completion_tokens) == (550, 75)
    call = fake.calls[0]
    assert call["enableTrace"] is True and call["agentId"] == "AGENT1" and call["agentAliasId"] == "TSTALIASID"
    assert call["sessionState"] == {"sessionAttributes": {"tenant": "nw"}}
    assert r.raw["session_id"] == call["sessionId"]


def test_bedrock_multi_turn_reuses_session_id():
    fake = FakeBedrock([{"chunk": {"bytes": b"Hi Ana."}}], [{"chunk": {"bytes": b"You are Ana."}}])
    agent = REGISTRY.build("system", {**BR, "client": fake})
    r = agent.run_conversation(["I'm Ana.", "Who am I?"])
    assert r.answer == "You are Ana."
    assert fake.calls[0]["sessionId"] == fake.calls[1]["sessionId"] == r.raw["session_id"]
    single = FakeBedrock([{"chunk": {"bytes": b"a"}}], [{"chunk": {"bytes": b"b"}}])
    a2 = REGISTRY.build("system", {**BR, "client": single})
    a2.run("q1"), a2.run("q2")
    assert single.calls[0]["sessionId"] != single.calls[1]["sessionId"]


def test_bedrock_stream_exception_failure_and_timeout():
    fake = FakeBedrock([{"chunk": {"bytes": b"partial"}},
                        {"throttlingException": {"message": "Rate exceeded"}}])
    with pytest.raises(RuntimeError, match="throttlingException: Rate exceeded"):
        REGISTRY.build("system", {**BR, "client": fake}).run("x")

    fake2 = FakeBedrock([{"trace": {"trace": {"failureTrace": {"failureReason": "Lambda timed out"}}}}])
    r = REGISTRY.build("system", {**BR, "client": fake2}).run("x")
    assert "Lambda timed out" in r.error

    def slow():
        time.sleep(0.05)
        yield {"chunk": {"bytes": b"late"}}
    with pytest.raises(TimeoutError):
        REGISTRY.build("system", {**BR, "client": FakeBedrock(slow()), "timeout": 0.01}).run("x")


def test_bedrock_return_control_with_handler():
    rc = {"returnControl": {"invocationId": "inv-1", "invocationInputs": [{"functionInvocationInput": {
        "actionGroup": "billing", "function": "refund",
        "parameters": [{"name": "amount", "type": "number", "value": "10"}]}}]}}
    fake = FakeBedrock([rc], [{"chunk": {"bytes": b"Refunded $10."}}])
    agent = REGISTRY.build("system", {**BR, "client": fake, "tool_handlers": {"refund": lambda a: "ok " + a["amount"]}})
    r = agent.run("refund me")
    assert r.answer == "Refunded $10." and r.trace[0].tool == "refund" and r.trace[0].result == "ok 10"
    ss = fake.calls[1]["sessionState"]
    assert ss["invocationId"] == "inv-1" and "inputText" not in fake.calls[1]
    assert ss["returnControlInvocationResults"][0]["functionResult"]["responseBody"]["TEXT"]["body"] == "ok 10"

    r2 = REGISTRY.build("system", {**BR, "client": FakeBedrock([rc])}).run("refund me")
    assert "refund" in r2.error and "tool_handlers" in r2.error


def test_bedrock_health_and_missing_sdk():
    control = NS(get_agent_alias=lambda agentId, agentAliasId: {
        "agentAlias": {"agentAliasName": "prod", "agentAliasStatus": "PREPARED"}})
    agent = REGISTRY.build("system", {**BR, "client": FakeBedrock(), "control_client": control})
    assert agent.health_check().ok
    if "boto3" in sys.modules:
        pytest.skip("boto3 installed")
    with pytest.raises(PluginDependencyError, match="evalkit\\[aws\\]"):
        REGISTRY.build("system", dict(BR))


# ====================================================================== HTTP fakes

class FakeHttpResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        return self._body if self._body is not None else json.loads(self.text)


class FakeSession:
    """Routes (method, url-substring) -> handler(json_body) -> FakeHttpResponse."""

    def __init__(self, routes):
        self.routes, self.calls, self.closed = routes, [], False

    def request(self, method, url, json=None, headers=None, timeout=None, **_):
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers or {}})
        for (m, frag), handler in self.routes:
            if m == method and frag in url:
                return handler(json, url)
        return FakeHttpResponse(404, text="no route")

    def close(self):
        self.closed = True


# ====================================================================== Vertex AI

def test_vertex_agent_engine_adk_stream_session_and_tokens():
    events = [
        {"author": "agent", "content": {"role": "model", "parts": [{"text": "Let me check."}]}},
        {"author": "agent", "content": {"role": "model", "parts": [
            {"function_call": {"id": "f1", "name": "get_weather", "args": {"city": "Paris"}}}]},
         "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 10}},
        {"author": "agent", "content": {"role": "user", "parts": [
            {"function_response": {"id": "f1", "name": "get_weather", "response": {"temp": 21}}}]}},
        {"author": "agent", "content": {"role": "model", "parts": [{"text": "It is 21°C in Paris."}]},
         "usage_metadata": {"promptTokenCount": 120, "candidatesTokenCount": 8},
         "grounding_metadata": {"grounding_chunks": [{"web": {"uri": "https://weather.example", "title": "Weather"}}]}},
    ]
    stream = "\n".join("data: " + json.dumps(e) for e in events)
    session = FakeSession([
        (("POST", ":streamQuery"), lambda body, url: FakeHttpResponse(200, text=stream)),
        (("POST", ":query"), lambda body, url: FakeHttpResponse(200, {"output": {"id": "sess-9", "user_id": "evalkit"}})),
        (("GET", "reasoningEngines/123"), lambda body, url: FakeHttpResponse(200, {"displayName": "Weather agent"})),
    ])
    agent = REGISTRY.build("system", {"type": "vertex_agent", "mode": "agent_engine", "project": "p",
                                      "location": "us-central1", "reasoning_engine_id": "123", "session": session})
    r = agent.run("Weather in Paris?")
    assert r.answer == "It is 21°C in Paris."
    assert r.tools_used == ["get_weather"] and r.trace[0].args == {"city": "Paris"}
    assert json.loads(r.trace[0].result) == {"temp": 21}
    assert (r.prompt_tokens, r.completion_tokens) == (220, 18)
    assert [d.id for d in r.retrieved] == ["https://weather.example"]
    call = session.calls[0]
    assert call["url"] == ("https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/"
                           "reasoningEngines/123:streamQuery?alt=sse")
    assert call["json"] == {"class_method": "stream_query", "input": {"user_id": "evalkit", "message": "Weather in Paris?"}}

    session.calls.clear()
    r2 = agent.run_conversation(["hi", "weather?"])
    assert session.calls[0]["json"]["class_method"] == "create_session"
    assert [c["json"]["input"].get("session_id") for c in session.calls[1:]] == ["sess-9", "sess-9"]
    assert r2.raw["session_id"] == "sess-9"
    assert agent.health_check().ok


def test_vertex_agent_engine_query_api_and_http_error():
    session = FakeSession([(("POST", ":query"), lambda body, url: FakeHttpResponse(200, {"output": {
        "input": "q", "output": "42 is the answer.",
        "intermediate_steps": [[{"tool": "calculator", "tool_input": {"expr": "6*7"}, "log": ""}, "42"]]}}))])
    agent = REGISTRY.build("system", {"type": "vertex_agent", "project": "p", "engine_api": "query",
                                      "reasoning_engine_id": "projects/p2/locations/europe-west4/reasoningEngines/77",
                                      "session": session})
    r = agent.run("6*7?")
    assert r.answer == "42 is the answer." and r.trace[0].tool == "calculator" and r.trace[0].result == "42"
    assert "europe-west4-aiplatform.googleapis.com/v1/projects/p2/" in session.calls[0]["url"]
    assert session.calls[0]["json"] == {"class_method": "query", "input": {"input": "6*7?"}}

    bad = FakeSession([(("POST", ":query"), lambda b, u: FakeHttpResponse(403, text="PERMISSION_DENIED"))])
    with pytest.raises(RuntimeError, match="HTTP 403"):
        REGISTRY.build("system", {"type": "vertex_agent", "project": "p", "engine_api": "query",
                                  "reasoning_engine_id": "1", "session": bad}).run("x")


def test_vertex_dialogflow_cx_detect_intent_session_reuse():
    def detect(body, url):
        return FakeHttpResponse(200, {"responseId": "r1", "queryResult": {
            "text": body["queryInput"]["text"]["text"],
            "responseMessages": [{"text": {"text": ["Your order ships Monday."]}}, {"payload": {"x": 1}}],
            "currentPage": {"displayName": "Start Page"},
            "match": {"matchType": "PLAYBOOK"},
            "generativeInfo": {"actionTracingInfo": {"actions": [
                {"playbookInvocation": {"playbook": "projects/p/locations/global/agents/a/playbooks/pb1",
                                        "displayName": "Orders"}},
                {"toolUse": {"tool": "projects/p/locations/global/agents/a/tools/t1", "displayName": "order_api",
                             "action": "getOrder", "inputActionParameters": {"id": "42"},
                             "outputActionParameters": {"status": "packed"}}},
                {"agentUtterance": {"text": "Your order ships Monday."}}]}},
            "dataStoreConnectionSignals": {
                "searchSnippets": [{"documentTitle": "Shipping", "documentUri": "gs://kb/shipping.html", "text": "Ships in 2 days"},
                                   {"documentTitle": "FAQ", "documentUri": "gs://kb/faq.html", "text": "FAQ"}],
                "citedSnippets": [{"searchSnippet": {"documentTitle": "Shipping", "documentUri": "gs://kb/shipping.html",
                                                     "text": "Ships in 2 days"}, "snippetIndex": 0}]}}})
    session = FakeSession([(("POST", ":detectIntent"), detect)])
    agent = REGISTRY.build("system", {"type": "vertex_agent", "mode": "dialogflow_cx", "project": "p",
                                      "location": "global", "agent_id": "a", "session": session})
    r = agent.run_conversation(["hi", "When does order 42 ship?"])
    assert r.answer == "Your order ships Monday."
    assert r.tools_used[:2] == ["playbook:Orders", "order_api"]
    assert r.trace[1].args == {"id": "42", "_action": "getOrder"} and "packed" in r.trace[1].result
    assert [d.id for d in r.retrieved] == ["gs://kb/shipping.html", "gs://kb/faq.html"]
    assert r.retrieved[0].metadata["cited"] is True and r.retrieved[1].metadata["cited"] is False
    urls = [c["url"] for c in session.calls]
    assert urls[0] == urls[1] and urls[0].startswith(
        "https://dialogflow.googleapis.com/v3/projects/p/locations/global/agents/a/sessions/evalkit-")
    assert session.calls[0]["json"]["queryParams"]["populateDataStoreConnectionSignals"] is True
    assert r.raw["match_type"] == "PLAYBOOK"


def test_vertex_config_errors_and_missing_google_auth():
    with pytest.raises(ConfigError, match="mode"):
        REGISTRY.build("system", {"type": "vertex_agent", "mode": "bogus", "project": "p", "session": FakeSession([])})
    with pytest.raises(ConfigError, match="agent_id"):
        REGISTRY.build("system", {"type": "vertex_agent", "mode": "dialogflow_cx", "project": "p",
                                  "session": FakeSession([])})
    try:
        import google.auth  # noqa: F401
        pytest.skip("google-auth installed")
    except ModuleNotFoundError:
        pass
    with pytest.raises(PluginDependencyError, match="evalkit\\[gcp\\]"):
        REGISTRY.build("system", {"type": "vertex_agent", "project": "p", "reasoning_engine_id": "1"})


def test_vertex_uses_auth_block_headers():
    session = FakeSession([(("POST", ":query"), lambda b, u: FakeHttpResponse(200, {"output": "ok"}))])
    agent = REGISTRY.build("system", {"type": "vertex_agent", "project": "p", "engine_api": "query",
                                      "reasoning_engine_id": "1", "session": session,
                                      "auth": {"type": "bearer", "token": "tok123"}})
    assert agent.run("x").answer == "ok"
    assert session.calls[0]["headers"]["Authorization"] == "Bearer tok123"


# ====================================================================== Copilot Studio (Direct Line)

class DirectLineServer:
    """Minimal Direct Line 3.0 simulation: bot replies appear after `delay_polls` GETs."""

    def __init__(self, replies_for, delay_polls=1, greeting=False):
        self.replies_for, self.delay_polls, self.greeting = replies_for, delay_polls, greeting
        self.activities, self.conversations, self.pending = [], [], []
        self.session = FakeSession([
            (("POST", "/tokens/generate"), self.generate),
            (("GET", "token.example"), lambda b, u: FakeHttpResponse(200, {"token": "endpoint-token", "expires_in": 3600})),
            (("POST", "/conversations/c1/activities"), self.post_activity),
            (("GET", "/conversations/c1/activities"), self.get_activities),
            (("POST", "/conversations"), self.start),
        ])

    def generate(self, body, url):
        return FakeHttpResponse(200, {"conversationId": "c1", "token": "conv-token", "expires_in": 3600})

    def start(self, body, url):
        self.conversations.append("c1")
        return FakeHttpResponse(201, {"conversationId": "c1", "token": "conv-token", "expires_in": 1800})

    def _add(self, act):
        act["id"] = f"c1|{len(self.activities):07d}"
        self.activities.append(act)
        return act["id"]

    def post_activity(self, body, url):
        aid = self._add(dict(body))
        if body.get("type") == "event" and self.greeting:
            self.pending.append([0, [{"type": "message", "from": {"id": "bot"}, "text": "Hello! I'm the bot.",
                                      "replyToId": aid}]])
        elif body.get("type") == "message":
            replies = [{**r, "from": {"id": "bot", "name": "Bot"}, "replyToId": aid}
                       for r in self.replies_for(body["text"])]
            self.pending.append([0, replies])
        return FakeHttpResponse(200, {"id": aid})

    def get_activities(self, body, url):
        for p in self.pending:
            p[0] += 1
            if p[0] >= self.delay_polls and p[1]:
                for r in p[1]:
                    self._add(r)
                p[1] = []
        wm = int(url.split("watermark=")[1]) if "watermark=" in url else 0
        return FakeHttpResponse(200, {"activities": self.activities[wm:], "watermark": str(len(self.activities))})


CS = {"type": "copilot_studio", "direct_line_secret": "secret-xyz", "poll_interval": 0, "settle_s": 0}


def test_copilot_studio_reply_citations_and_events():
    def replies(text):
        return [
            {"type": "typing"},
            {"type": "event", "name": "DynamicPlanStepTriggered", "value": {"taskDialogId": "LookupOrder",
                                                                            "arguments": {"order": "42"}}},
            {"type": "event", "name": "DynamicPlanStepFinished", "value": {"taskDialogId": "LookupOrder",
                                                                           "observation": "shipped"}},
            {"type": "message", "text": "Order 42 shipped [1].",
             "entities": [{"type": "https://schema.org/Message", "citation": [
                 {"@type": "Claim", "position": 1, "appearance": {"@type": "DigitalDocument", "name": "Orders FAQ",
                                                                  "url": "https://contoso.com/faq", "abstract": "Shipping info"}}]}]},
            {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive",
                                                 "content": {"type": "AdaptiveCard", "body": [
                                                     {"type": "TextBlock", "text": "Tracking: 1Z999"}]}}]},
        ]
    server = DirectLineServer(replies)
    agent = REGISTRY.build("system", {**CS, "session": server.session})
    r = agent.run("Where is order 42?")
    assert r.answer == "Order 42 shipped [1].\nTracking: 1Z999"
    assert r.tools_used == ["LookupOrder"] and r.trace[0].args == {"order": "42"} and r.trace[0].result == "shipped"
    assert [d.id for d in r.retrieved] == ["https://contoso.com/faq"] and r.retrieved[0].text == "Shipping info"
    assert r.raw["conversation_id"] == "c1"
    calls = server.session.calls
    assert calls[0]["url"].endswith("/tokens/generate") and calls[0]["headers"]["Authorization"] == "Bearer secret-xyz"
    assert calls[1]["headers"]["Authorization"] == "Bearer conv-token"
    sent = [a for a in server.activities if a.get("from", {}).get("id") == "evalkit-user"]
    assert sent[0]["text"] == "Where is order 42?" and sent[0]["type"] == "message"
    assert agent.health_check().ok


def test_copilot_studio_multi_turn_same_conversation_with_greeting():
    server = DirectLineServer(lambda text: [{"type": "message", "text": f"echo: {text}"}], greeting=True)
    agent = REGISTRY.build("system", {**CS, "session": server.session, "send_start_event": True})
    r = agent.run_conversation(["first", "second"])
    assert r.answer == "echo: second"
    assert server.conversations == ["c1"]           # one conversation for all turns
    gets = [c["url"] for c in server.session.calls if c["method"] == "GET"]
    assert any("watermark=" in u for u in gets)


def test_copilot_studio_token_endpoint_timeout_and_errors():
    server = DirectLineServer(lambda text: [])      # bot never answers
    agent = REGISTRY.build("system", {"type": "copilot_studio", "session": server.session, "poll_interval": 0.01,
                                      "timeout": 0.05, "token_endpoint": "https://token.example/directline/token"})
    with pytest.raises(TimeoutError, match="did not reply"):
        agent.run("hello?")
    assert server.session.calls[0]["method"] == "GET" and "token.example" in server.session.calls[0]["url"]
    assert server.session.calls[1]["headers"]["Authorization"] == "Bearer endpoint-token"

    denied = FakeSession([(("POST", "/tokens/generate"), lambda b, u: FakeHttpResponse(403, text="Forbidden"))])
    with pytest.raises(RuntimeError, match="HTTP 403"):
        REGISTRY.build("system", {**CS, "session": denied}).run("x")
    with pytest.raises(ConfigError, match="direct_line_secret"):
        REGISTRY.build("system", {"type": "copilot_studio", "session": denied})
