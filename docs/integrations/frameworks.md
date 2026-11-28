# Agent frameworks and MCP tool servers

evalkit can evaluate agents built with LangChain/LangGraph, LlamaIndex, CrewAI and
Semantic Kernel in-process, and can evaluate an MCP tool server by driving it with a
model of your choice. Each adapter turns one test question into a standard
`AgentResult` (answer, tool-call trace, retrieved documents, token usage, latency).

| type | class | status |
|---|---|---|
| `mcp` | `evalkit.adapters.mcp_agent:McpToolAgent` | **verified** end to end against a real MCP server (this repo's `mcp_server.py`) with a scripted model |
| `langchain` | `evalkit.adapters.frameworks:LangChainAgent` | mock-tested with duck-typed fakes; not yet verified against the real framework packages |
| `llamaindex` | `evalkit.adapters.frameworks:LlamaIndexAgent` | mock-tested with duck-typed fakes; not yet verified against the real framework packages |
| `crewai` | `evalkit.adapters.frameworks:CrewAIAgent` | mock-tested with duck-typed fakes; not yet verified against the real framework packages |
| `semantic_kernel` | `evalkit.adapters.frameworks:SemanticKernelAgent` | mock-tested with duck-typed fakes; not yet verified against the real framework packages |

Tests: `tests/test_frameworks.py`.

All four framework adapters import nothing from the framework themselves: they
call the object you point them at, so the framework only needs to be installed
in the environment where your agent already runs. `target` is always
`module:attribute`, naming either the object or a zero-argument function that
builds it (called once per run).

## Data flow (all adapters)

evalkit runs your agent **in the same Python process**. evalkit itself sends
nothing anywhere: every network call (LLM provider, vector database, tool APIs)
is made by your agent with your agent's own configuration and credentials,
exactly as in production. What evalkit stores locally is the `AgentResult`:
the answer, tool names/arguments/results (each result capped at 20,000
characters), retrieved document ids/text/scores/simple metadata, and token
counts. These land in the run's results file, so treat that file with the same
sensitivity as your agent's logs. Judges, if configured, send question, answer
and context to the judge provider you configured.

The `mcp` adapter is different: evalkit itself plays the agent, so it sends the
question, the server's tool list and every tool result to the model provider you
configure (see below).

---

## LangChain / LangGraph (`langchain`)

```yaml
system:
  type: langchain
  target: my_app.agent:agent_executor     # any Runnable with .invoke()
  input_key: input                        # payload = {input_key: question}
  output_key: output                      # optional; dotted path into a dict output
  inputs: { chat_history: [] }            # optional; extra static input keys
  config: { recursion_limit: 25 }         # optional; merged into the RunnableConfig
  retriever_hook: my_app.agent:set_search # optional; enables prompt-injection tests
```

LangGraph message graphs (e.g. `create_react_agent`):

```yaml
system:
  type: langchain
  target: my_app.graph:build_graph        # returns graph.compile()
  input_format: messages                  # payload = {"messages": [("user", question)]}
```

Options: `target`, `input_key` (default `input`), `output_key`, `input_format`
(`dict` default, `messages`, or `string` to pass the bare question),
`history_key`, `inputs`, `config`, `retriever_hook`.

What is captured, via a callback handler (a `BaseCallbackHandler` subclass when
`langchain_core` is importable) added to `config["callbacks"]`:

- **trace**: `on_tool_start` / `on_tool_end` / `on_tool_error` → one `ToolCall`
  per tool run (tool name, structured inputs when given, else the input string, result).
- **retrieved**: `on_retriever_end` documents (id = `Document.id`, else metadata
  `id`/`source`/`file_name`; score from metadata `score`/`relevance_score`).
  If no retriever callback fired, `source_documents` / `context` from a dict output are used.
- **tokens**: `on_llm_end`: `AIMessage.usage_metadata` (input/output tokens) when
  present, otherwise `llm_output["token_usage"]`; never both for the same call.
- **answer**: `output_key` if set; else the last message of `messages`, or the
  `output`/`answer`/`result`/`text`/`response` key, or the message/string itself.
  Content-block lists are joined to text.

Multi-turn: with `input_format: messages` the full history is passed as messages
each turn; with `history_key: chat_history` the history is passed as
`[("human", ...), ("ai", ...)]` under that key. Otherwise evalkit's default
transcript replay is used. (A LangGraph checkpointer is not needed and not used.)

Prompt-injection tests: only when `retriever_hook` names a function that swaps
your agent's search backend, as for the `python` adapter.

`health_check` only confirms the target loaded; it does not call the model.

## LlamaIndex (`llamaindex`)

```yaml
system:
  type: llamaindex
  target: my_app.rag:query_engine         # QueryEngine, ChatEngine, or agent with .chat()
  mode: auto                              # auto | query | chat
  token_counter: my_app.rag:token_counter # optional TokenCountingHandler
```

Options: `target`, `mode` (`auto` uses `.query()` when present, else `.chat()`),
`token_counter`.

What is captured:

- **answer**: `response.response` (or `str(response)`).
- **retrieved**: `response.source_nodes` → id = `node.node_id` (else metadata
  `file_name`), text = `node.get_content()`, score, simple metadata.
- **trace**: `response.sources` (`ToolOutput`) → tool name, `raw_input`
  (LlamaIndex's `{"args", "kwargs"}` is flattened to the kwargs), content.
- **tokens**: only if you pass the `TokenCountingHandler` you attached to
  `Settings.callback_manager`; evalkit calls `reset_counts()` before each call
  and reads `prompt_llm_token_count` / `completion_llm_token_count`. Otherwise `None`.
  Because the counter is shared, do not run cases concurrently when using it.

Multi-turn: chat engines and agents are used natively: `reset()` once, then one
`.chat()` per turn. Single-turn cases in chat mode call `reset()` first so cases
do not leak into each other.

## CrewAI (`crewai`)

```yaml
system:
  type: crewai
  target: my_app.crew:build_crew          # a Crew, or a function returning one
  input_key: question                     # kickoff(inputs={question: <test input>})
  inputs: { audience: employees }         # optional static inputs
  fresh_copy: true                        # default; crew.copy() per case
```

Options: `target`, `input_key` (default `input`), `inputs`, `fresh_copy`.

What is captured:

- **answer**: `result.raw` (else `str(result)`).
- **tokens**: `result.token_usage.prompt_tokens` / `completion_tokens`;
  `total_tokens` goes to `raw`.
- **trace**: one entry per `tasks_output` item: tool = agent role, args =
  `{"description": task description}`, result = the task's raw output. Individual
  tool calls made inside CrewAI agents are not in `CrewOutput`, so they are not
  captured.

Multi-turn: evalkit's default transcript replay (crews have no chat session).

## Semantic Kernel (`semantic_kernel`)

Kernel function:

```yaml
system:
  type: semantic_kernel
  target: my_app.sk:build_kernel          # a Kernel (or factory)
  function: policies.answer               # plugin_name.function_name
  input_arg: input                        # the KernelArguments name for the question
```

Agent (`ChatCompletionAgent` and other SK agents with `invoke` / `get_response`):

```yaml
system:
  type: semantic_kernel
  target: my_app.sk:agent
```

Options: `target`, `function` (or `plugin_name` + `function_name`), `input_arg`
(default `input`).

SK is async. evalkit runs every call on one private event loop in a worker
thread, so it works from plain scripts, from inside an already-running loop
(Jupyter, async apps), and keeps SK's async HTTP clients on a single loop across
cases. `close()` stops that loop.

What is captured:

- **trace**: `FunctionCallContent` / `FunctionResultContent` items (matched by
  call id) found in returned messages and, for agents, in intermediate messages
  delivered through `on_intermediate_message`. Tool name = `plugin-function`.
- **tokens**: `metadata["usage"]` (`prompt_tokens` / `completion_tokens`) on
  messages and on the `FunctionResult`, when present.
- **answer**: the last message's content, or the function result's value.

Multi-turn: agents use their `AgentThread` natively (the thread returned by turn
1 is passed to turn 2); kernel functions use evalkit's default transcript replay.

---

## MCP tool server (`mcp`)

evalkit runs a standard tool-calling loop: it lists the server's tools, gives
them to the model in OpenAI function-calling format, executes each tool call
through MCP, feeds the results back, and repeats until the model answers or
`max_steps` model calls are used.

```yaml
system:
  type: mcp
  server:                                  # stdio: evalkit starts the process
    command: python
    args: [mcp_server.py]
    env: { RETRIEVAL_METHOD: bm25 }        # optional extra environment variables
    cwd: .                                 # optional
  model:
    type: openai_compatible                # or azure_openai
    base_url: https://api.groq.com/openai/v1
    api_key: ${GROQ_API_KEY}
  model_name: llama-3.3-70b-versatile
  system_prompt: Answer from the policy tools and cite the source file.
  max_steps: 6
  tools: [search_policies, read_document]  # optional allow-list
```

Remote servers:

```yaml
  server: { url: https://tools.example.com/mcp, headers: { Authorization: "Bearer ${MCP_TOKEN}" } }  # streamable HTTP
  server: { url: https://tools.example.com/sse, transport: sse }                                      # legacy SSE
```

Options: `server` (`command`, `args`, `env`, `cwd` | `url`, `headers`,
`transport`; `connect_timeout`, default 60 s), `model` (a model-provider block,
`type` defaults to `openai_compatible`), `model_name` (falls back to
`model.model` / the provider's `default_model`), `system_prompt`, `max_steps`
(default 6), `tools`, `temperature` (default 0), `tool_timeout` (default 120 s).

What is captured:

- **trace**: every tool call: name, parsed JSON arguments, text result (non-text
  content blocks are summarised as `[image]`, `[resource <uri>]`). Invalid
  arguments, unknown tool names and tool errors are recorded in the trace and
  sent back to the model, as a real agent would; `raw.tool_errors` counts them.
- **tokens**: summed `usage` over every model call in the loop.
- **error**: set when no final answer arrives within `max_steps`.
- Multi-turn: native: the whole message history, including tool calls and
  results, carries over between turns.

Session handling: one MCP session is opened on first use and reused for every
case (the server process is started once); `close()` ends it and the server.
Tool calling uses the provider's `.client.chat.completions.create`, which
`openai_compatible` and `azure_openai` expose. Providers without native tool
calling (anthropic, bedrock, vertex, ollama's native API) raise a `ConfigError`;
for Ollama, use `openai_compatible` with `base_url: http://localhost:11434/v1`.

Data flow: the question, system prompt, tool schemas and tool results go to the
configured model endpoint. Tool calls go to the MCP server (a local process for
stdio, or the URL you give). Nothing else leaves the machine.

Works with the MCP Python SDK 2.x (tested with 2.3.0) and handles the 1.x
names (`streamablehttp_client`, `inputSchema`, `isError`).

## Open questions for the framework adapters

These follow the frameworks' documented APIs but have not been run against the
real packages yet:

- LangChain: tool `inputs` are only passed to `on_tool_start` by recent
  `langchain_core`; older versions give only the input string (captured as `{"input": ...}`).
- LlamaIndex: the newer workflow agents (`FunctionAgent`, `AgentWorkflow`, async
  `.run()`) are not supported; use a query/chat engine or a `.chat()` agent.
- CrewAI: `Crew.copy()` is used per case by default; set `fresh_copy: false` if
  your crew does not copy cleanly.
- Semantic Kernel: requires the agent API with `invoke(messages=..., thread=...)`
  (SK Python 1.2x+); the older `invoke(chat_history)` API is not supported.
