"""Framework adapters (duck-typed fakes) and the MCP tool agent (real MCP server,
scripted model). No network, no API keys."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from evalkit.adapters.frameworks import CrewAIAgent, LangChainAgent, LlamaIndexAgent, SemanticKernelAgent
from evalkit.errors import ConfigError
from evalkit.plugins import REGISTRY
from evalkit.types import RetrievedDoc

ROOT = Path(__file__).resolve().parent.parent


# ---------- LangChain fakes ----------
class FakeChain:
    """Behaves like an AgentExecutor: fires callbacks, returns {"output": ...}."""

    def __init__(self):
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append(payload)
        cb = config["callbacks"][-1]
        cb.on_retriever_end([NS(page_content="Hotels: $250/night in London.", id="travel.md",
                                metadata={"source": "travel.md", "score": 0.9})], run_id="r1")
        cb.on_tool_start({"name": "calculator"}, "4*250", run_id="t1", inputs={"expression": "4*250"})
        cb.on_tool_end(NS(content="1000"), run_id="t1")
        cb.on_llm_end(NS(generations=[], llm_output={"token_usage": {"prompt_tokens": 30, "completion_tokens": 7}}))
        cb.on_llm_end(NS(generations=[[NS(message=NS(usage_metadata={"input_tokens": 5, "output_tokens": 2}))]],
                         llm_output={"token_usage": {"prompt_tokens": 999, "completion_tokens": 999}}))
        return {"input": payload, "output": "You can claim $1000."}


def fake_chain():  # zero-arg factory, used through the registry
    return FakeChain()


class FakeGraph:
    """Behaves like a compiled LangGraph message graph."""

    def __init__(self):
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append(payload)
        n = len([m for m in payload["messages"] if m[0] == "user"])
        return {"messages": payload["messages"] + [NS(content=[{"type": "text", "text": f"reply {n}"}])]}


_hooked = []


def record_retriever(r):
    _hooked.append(r)


def test_langchain_captures_tools_docs_tokens():
    chain = FakeChain()
    r = LangChainAgent(target=chain).run("How much for 4 nights?")
    assert r.answer == "You can claim $1000."
    assert chain.calls == [{"input": "How much for 4 nights?"}]
    assert [(t.tool, t.args, t.result) for t in r.trace] == [("calculator", {"expression": "4*250"}, "1000")]
    assert r.retrieved[0].id == "travel.md" and r.retrieved[0].score == 0.9
    # llm_output on the first call, usage_metadata preferred on the second (no double count)
    assert (r.prompt_tokens, r.completion_tokens) == (35, 9)
    assert r.error is None and r.latency_s is not None


def test_langchain_through_registry_with_factory():
    agent = REGISTRY.build("system", {"type": "langchain", "target": "tests.test_frameworks:fake_chain"})
    assert isinstance(agent, LangChainAgent) and isinstance(agent.runnable, FakeChain)
    assert agent.run("q").answer == "You can claim $1000."
    assert not agent.supports_retriever_injection
    with pytest.raises(NotImplementedError):
        agent.set_retriever(None)


def test_langchain_messages_graph_multi_turn():
    graph = FakeGraph()
    agent = LangChainAgent(target=graph, input_format="messages")
    assert agent.run("hi").answer == "reply 1"
    r = agent.run_conversation(["first", "second"])
    assert r.answer == "reply 2"
    assert graph.calls[-1]["messages"] == [("user", "first"), ("assistant", "reply 1"), ("user", "second")]
    assert [t["user"] for t in r.raw["turns"]] == ["first", "second"]


def test_langchain_output_key_and_errors():
    class Boom:
        def invoke(self, payload, config=None):
            raise RuntimeError("model down")

    r = LangChainAgent(target=Boom()).run("q")
    assert r.answer == "" and "model down" in r.error
    nested = NS(invoke=lambda p, config=None: {"result": {"text": "deep"}})
    assert LangChainAgent(target=nested, output_key="result.text").run("q").answer == "deep"
    with pytest.raises(ConfigError):
        LangChainAgent(target=nested, input_format="bogus")
    with pytest.raises(ConfigError):
        LangChainAgent(target=NS(nothing=1))


def test_langchain_retriever_hook():
    _hooked.clear()
    agent = LangChainAgent(target=FakeChain(), retriever_hook="tests.test_frameworks:record_retriever")
    assert agent.supports_retriever_injection
    agent.set_retriever("poisoned")
    assert _hooked == ["poisoned"]


# ---------- LlamaIndex fakes ----------
def _li_response(text):
    node = NS(node=NS(node_id="n-1", metadata={"file_name": "leave.md"}, get_content=lambda: "25 days leave"),
              score=0.8)
    tool = NS(tool_name="search", raw_input={"args": (), "kwargs": {"query": "leave"}}, content="25 days")
    return NS(response=text, source_nodes=[node], sources=[tool])


class FakeQueryEngine:
    def query(self, q):
        return _li_response(f"answer to {q}")


class FakeChatEngine:
    def __init__(self):
        self.history, self.resets = [], 0

    def chat(self, msg):
        self.history.append(msg)
        return _li_response(f"turn {len(self.history)}")

    def reset(self):
        self.history, self.resets = [], self.resets + 1


class FakeTokenCounter:
    prompt_llm_token_count = completion_llm_token_count = 0

    def reset_counts(self):
        self.prompt_llm_token_count, self.completion_llm_token_count = 11, 4  # as if counted during the call


def test_llamaindex_query_engine():
    r = LlamaIndexAgent(target=FakeQueryEngine(), token_counter=FakeTokenCounter()).run("leave?")
    assert r.answer == "answer to leave?"
    assert r.retrieved == [RetrievedDoc(id="n-1", text="25 days leave", score=0.8, metadata={"file_name": "leave.md"})]
    assert (r.trace[0].tool, r.trace[0].args, r.trace[0].result) == ("search", {"query": "leave"}, "25 days")
    assert (r.prompt_tokens, r.completion_tokens) == (11, 4)
    assert LlamaIndexAgent(target=FakeQueryEngine()).run("x").prompt_tokens is None


def test_llamaindex_chat_engine_native_memory():
    engine = FakeChatEngine()
    agent = LlamaIndexAgent(target=engine)
    assert agent.mode == "chat"
    r = agent.run_conversation(["a", "b", "c"])
    assert r.answer == "turn 3" and engine.history == ["a", "b", "c"] and engine.resets == 1
    assert len(r.trace) == 3 and len(r.retrieved) == 3
    assert agent.run("fresh").answer == "turn 1"  # each single-turn case starts a new conversation


# ---------- CrewAI fake ----------
class FakeCrew:
    def __init__(self):
        self.kicked, self.copies = [], 0

    def copy(self):
        self.copies += 1
        return self

    def kickoff(self, inputs):
        self.kicked.append(inputs)
        tasks = [NS(agent="Researcher", description="Find the policy", raw="Found leave.md"),
                 NS(agent="Writer", description="Write answer", raw="25 days")]
        return NS(raw="Employees get 25 days.", tasks_output=tasks,
                  token_usage=NS(prompt_tokens=100, completion_tokens=20, total_tokens=120))


def test_crewai_kickoff():
    crew = FakeCrew()
    r = CrewAIAgent(target=crew, input_key="question", inputs={"tone": "brief"}).run("leave?")
    assert crew.kicked == [{"tone": "brief", "question": "leave?"}] and crew.copies == 1
    assert r.answer == "Employees get 25 days."
    assert [(t.tool, t.args["description"], t.result) for t in r.trace] == [
        ("Researcher", "Find the policy", "Found leave.md"), ("Writer", "Write answer", "25 days")]
    assert (r.prompt_tokens, r.completion_tokens, r.raw["total_tokens"]) == (100, 20, 120)


# ---------- Semantic Kernel fakes ----------
def _call(id_, name, args):
    return NS(content_type="function_call", id=id_, function_name=name, plugin_name="policies", arguments=args)


def _result(id_, name, result):
    return NS(content_type="function_result", id=id_, function_name=name, plugin_name="policies", result=result)


class FakeKernel:
    def __init__(self):
        self.calls = []

    async def invoke(self, plugin_name=None, function_name=None, **kwargs):
        await asyncio.sleep(0)
        self.calls.append((plugin_name, function_name, kwargs))
        msg = NS(content="25 days", items=[_call("c1", "search", '{"query": "leave"}'),
                                            _result("c1", "search", "leave.md: 25 days")],
                 metadata={"usage": NS(prompt_tokens=12, completion_tokens=3)})
        return NS(value=[msg], metadata={})


class FakeSkAgent:
    instructions = "You answer policy questions."

    def __init__(self):
        self.threads = []

    async def get_response(self, messages=None, thread=None):  # pragma: no cover - invoke is used
        raise AssertionError

    async def invoke(self, messages=None, thread=None, on_intermediate_message=None):
        self.threads.append(thread)
        thread = thread or NS(history=[])
        thread.history.append(messages)
        if on_intermediate_message:
            await on_intermediate_message(NS(content="", items=[_call("x", "search", {"query": messages})],
                                             metadata={}))
            await on_intermediate_message(NS(content="", items=[_result("x", "search", "hit")], metadata={}))
        final = NS(content=f"answer {len(thread.history)}", items=[],
                   metadata={"usage": NS(prompt_tokens=10, completion_tokens=2)})
        yield NS(message=final, thread=thread)


def test_semantic_kernel_function():
    kernel = FakeKernel()
    agent = SemanticKernelAgent(target=kernel, function="policies.answer")
    r = agent.run("leave?")
    assert kernel.calls == [("policies", "answer", {"input": "leave?"})]
    assert r.answer == "25 days"
    assert [(t.tool, t.args, t.result) for t in r.trace] == [
        ("policies-search", {"query": "leave"}, "leave.md: 25 days")]
    assert (r.prompt_tokens, r.completion_tokens) == (12, 3)
    agent.close()
    with pytest.raises(ConfigError):
        SemanticKernelAgent(target=kernel)  # a kernel needs a function name


def test_semantic_kernel_agent_thread_and_running_loop():
    sk = FakeSkAgent()
    agent = SemanticKernelAgent(target=sk)
    assert agent.is_agent
    r = agent.run_conversation(["one", "two"])
    assert r.answer == "answer 2"
    assert sk.threads[0] is None and sk.threads[1] is not None  # thread carried across turns
    assert [t.result for t in r.trace] == ["hit", "hit"] and r.trace[0].args == {"query": "one"}
    assert (r.prompt_tokens, r.completion_tokens) == (20, 4)

    async def from_async_code():  # caller already inside an event loop
        return agent.run("again")
    assert asyncio.run(from_async_code()).answer == "answer 1"
    agent.close()


# ---------- MCP tool agent: real MCP server, scripted model ----------
def _tool_call(id_, name, args):
    return NS(id=id_, type="function", function=NS(name=name, arguments=json.dumps(args)))


class ScriptedClient:
    """OpenAI-style client: .chat.completions.create returns scripted replies."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=str)))  # snapshot (messages list mutates)
        msg = self.replies.pop(0)
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=50, completion_tokens=10))

    def close(self):
        pass


def _server():
    return {"command": sys.executable, "args": [str(ROOT / "mcp_server.py")], "cwd": str(ROOT)}


def test_mcp_tool_agent_end_to_end():
    client = ScriptedClient([
        NS(content=None, tool_calls=[_tool_call("c1", "calculator", {"expression": "4*320 + 4*75"}),
                                     _tool_call("c2", "list_documents", {})]),
        NS(content=None, tool_calls=[_tool_call("c3", "search_policies", {"query": "hotel London", "top_k": 1})]),
        NS(content="The total is 1580.", tool_calls=None),
    ])
    agent = REGISTRY.build("system", {"type": "mcp", "server": _server(), "system_prompt": "Use tools.",
                                      "model": {"type": "openai_compatible", "client": client},
                                      "model_name": "scripted"})
    try:
        r = agent.run("What is 4 nights at 320 plus 4 days at 75?")
        assert r.error is None, r.error
        assert r.answer == "The total is 1580."
        assert [t.tool for t in r.trace] == ["calculator", "list_documents", "search_policies"]
        assert "1580" in r.trace[0].result
        assert ".md" in r.trace[1].result and r.trace[2].result
        assert (r.prompt_tokens, r.completion_tokens) == (150, 30)
        # the model saw the server's real tools in OpenAI format, and the tool results came back
        first = client.requests[0]
        names = {t["function"]["name"] for t in first["tools"]}
        assert {"search_policies", "calculator", "read_document", "list_documents"} <= names
        calc = next(t for t in first["tools"] if t["function"]["name"] == "calculator")
        assert "expression" in calc["function"]["parameters"]["properties"]
        assert first["messages"][0] == {"role": "system", "content": "Use tools."}
        tool_msgs = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
        assert tool_msgs[0]["tool_call_id"] == "c1" and "1580" in tool_msgs[0]["content"]

        # same session reused for the next case; unknown tools and max_steps are handled
        client.replies += [NS(content=None, tool_calls=[_tool_call("u1", "rm_rf", {})]),
                           NS(content="Sorry.", tool_calls=None)]
        r2 = agent.run("delete everything")
        assert r2.answer == "Sorry." and "unknown tool" in r2.trace[0].result
        assert agent.health_check().ok
    finally:
        agent.close()


def test_mcp_tool_agent_multi_turn_and_max_steps():
    loop_forever = [NS(content=None, tool_calls=[_tool_call(f"k{i}", "calculator", {"expression": "1+1"})])
                    for i in range(2)]
    client = ScriptedClient([NS(content="Hi.", tool_calls=None), NS(content="Still here.", tool_calls=None)]
                            + loop_forever)
    agent = REGISTRY.build("system", {"type": "mcp", "server": _server(), "max_steps": 2,
                                      "model": {"type": "openai_compatible", "client": client, "model": "m"}})
    try:
        r = agent.run_conversation(["hello", "again"])
        assert r.answer == "Still here." and [t["assistant"] for t in r.raw["turns"]] == ["Hi.", "Still here."]
        assert [m["role"] for m in client.requests[1]["messages"]] == ["user", "assistant", "user"]
        r2 = agent.run("loop")
        assert "max_steps=2" in r2.error and len(r2.trace) == 2
    finally:
        agent.close()


def test_mcp_tool_agent_config_errors():
    with pytest.raises(ConfigError, match="server"):
        REGISTRY.build("system", {"type": "mcp", "server": {}, "model": {"type": "openai_compatible", "client": ScriptedClient([])},
                                  "model_name": "m"})

    class TextOnly:
        def complete(self, messages, model, **kw):
            return None

    from evalkit.adapters.mcp_agent import McpToolAgent
    with pytest.raises(ConfigError, match="native tool calling"):
        McpToolAgent(server=_server(), model=TextOnly(), model_name="m")
