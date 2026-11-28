# Hosted agent systems

These adapters let evalkit test an agent that runs on a cloud agent service.
Put one in the `system:` block of your config, the same way you would use
`type: http` or `type: python`. Each adapter sends a test question to the
agent through the vendor's own API and turns the reply into evalkit's standard
`AgentResult`:

| field | meaning |
|---|---|
| `answer` | the agent's final reply text |
| `trace` | tool calls (`tool`, `args`, `result`) |
| `retrieved` | documents the agent retrieved or cited (`id`, `text`, `score`, `metadata.cited`) |
| `prompt_tokens` / `completion_tokens` | token usage, when the service reports it |
| `latency_s` | wall-clock time for the test case |
| `error` | set when the run failed, was cancelled, or needs something evalkit cannot provide |
| `raw` | a short summary with ids (thread, run, session, conversation) for debugging |

Multi-turn test cases (`turns:`) use the service's own session: the same thread,
session or conversation is reused for every turn. The result holds the last
turn's answer, with tool calls, documents and tokens added up across all turns.
Each new test case starts a new session.

Every adapter waits for a limited time only. If the service has not answered
within `timeout` seconds, the adapter raises `TimeoutError`. Where the API
allows it, the adapter also cancels the run first.

`evalkit plugins health <config>` runs each adapter's `health_check()`. Where the API allows,
this is a cheap metadata lookup (get the agent, assistant or alias) instead of
a full question.

**Function tools that run on your side.** OpenAI, Azure and Bedrock (return of
control) can hand a function call back to the caller. YAML cannot hold Python
functions, so for these calls you build the adapter in Python and pass
`tool_handlers={"tool_name": fn}`, where `fn(args: dict) -> str`. If no handler
is given, the call still shows up in the trace and the result carries an
`error` that explains what is missing. Tools that run on the service itself
(file search, Azure AI Search, Bing, knowledge bases, code interpreter) need no
handler.

---

## OpenAI Responses API (`openai_responses`)

**Status: mock-tested (implemented against the documented API; not yet verified against a live service)**

API: `POST /v1/responses` (`client.responses.create`). Multi-turn conversations
chain the turns with `previous_response_id`.

```yaml
system:
  type: openai_responses
  model: gpt-4.1
  instructions: "You are the Northwind support agent."
  tools:
    - { type: file_search, vector_store_ids: [vs_abc123] }
    - { type: web_search_preview }
  api_key: ${OPENAI_API_KEY}
  timeout: 120
```

Options: `model` (or a stored `prompt: {id: pmpt_..., version: "2"}`),
`instructions`, `tools`, `include` (default: `file_search_call.results` when
file_search is configured), `temperature`, `max_output_tokens`, `reasoning`,
`store` (default `true`), `extra_params` (any other `responses.create`
argument), `api_key`, `base_url`, `organization`, `project`, `timeout`,
`max_tool_rounds`.

Credentials: `api_key`, or the `OPENAI_API_KEY` environment variable.
`base_url` points the adapter at an OpenAI-compatible gateway.

What it captures:
- answer: `output_text`
- trace: `file_search`, `web_search`, `code_interpreter`, MCP and function calls
- retrieved: file_search chunks (with score) plus `file_citation` and `url_citation` annotations
- tokens: `usage.input_tokens` and `usage.output_tokens`
- raw: response ids and status

With `store: false`, multi-turn cases resend the conversation history because
`previous_response_id` cannot be used.

Data that leaves your environment: test questions, plus any instructions and
tool settings you configure, are sent to the OpenAI API (or to the `base_url`
you set).

## OpenAI Assistants API (`openai_assistants`)

**Status: mock-tested (implemented against the documented API; not yet verified against a live service)**

API: Assistants v2. The adapter creates a thread, posts a message, starts a run,
and polls `runs.retrieve` until the run finishes. It then reads the reply with
`messages.list(run_id=...)` and the tool calls with `runs.steps.list`. OpenAI
has announced that the Assistants API is deprecated in favor of the Responses
API.

```yaml
system:
  type: openai_assistants
  assistant_id: asst_abc123
  api_key: ${OPENAI_API_KEY}
  timeout: 120
  poll_interval: 1.0
  delete_threads: true
```

Options: `assistant_id`, `additional_instructions`, `model` (overrides the model
for each run), `api_key`, `base_url`, `organization`, `project`, `timeout`,
`poll_interval`, `delete_threads`, `include_search_content` (asks the run steps
to include the text of file_search chunks), `strip_citation_markers` (removes
`【4:0†source】` markers from the answer).

What it captures:
- trace: function, `file_search` and `code_interpreter` calls from the run steps
- retrieved: file_search results with their text and score, plus file citations
- tokens: `run.usage`
- raw: `thread_id`, `run_ids`, status

A failed, expired or cancelled run sets `error` to the run's `last_error`.

Data that leaves your environment: test questions go to the OpenAI API, where
they are stored in a thread. The thread is deleted at the end of the test case
unless you set `delete_threads: false`.

## Azure AI Foundry Agent Service (`azure_foundry`)

**Status: mock-tested (implemented against the documented API; not yet verified against a live service)**

API: `azure-ai-projects` 1.x / `azure-ai-agents` 1.x. The client is
`AIProjectClient(endpoint, credential).agents`. The adapter calls
`threads.create`, then `messages.create`, then `runs.create` and polls
`runs.get` until the run finishes. It then reads `messages.list` and
`run_steps.list`. Older preview SDKs used flat method names
(`create_thread`, `create_message`, ...); the adapter falls back to those
automatically.

```yaml
system:
  type: azure_foundry
  endpoint: https://<resource>.services.ai.azure.com/api/projects/<project>
  agent_id: asst_abc123
  timeout: 180
```

Options: `endpoint`, `agent_id`, `connection_string` (old hub-based projects
only), `additional_instructions`, `timeout`, `poll_interval`, `delete_threads`,
`use_create_and_process` (use the SDK's `runs.create_and_process` instead of
evalkit's polling; this has no time limit), `include_search_content`,
`strip_citation_markers`, `auto_approve_tools` (approve MCP tool calls that ask
for approval).

Credentials: `DefaultAzureCredential` by default. It picks up a managed
identity, workload identity, `az login`, or the `AZURE_CLIENT_ID`,
`AZURE_TENANT_ID` and `AZURE_CLIENT_SECRET` environment variables. To use a
specific identity, set `managed_identity_client_id`. For a service principal,
set `tenant_id`, `client_id` and `client_secret`, preferably with
`${secret:...}`. The identity needs the Azure AI User role (or a role with
agent permissions) on the project.

What it captures:
- trace: one entry per tool call in the run steps, including `file_search`,
  `azure_ai_search` (args `{query}`), `bing_grounding` (args `{request_url, query}`),
  `sharepoint_grounding`, `openapi`, `connected_agent`, `mcp`, `code_interpreter`
  and function calls (with their output)
- retrieved: file_search chunks, plus `url_citation` and `file_citation` annotations on the reply
- tokens: `run.usage.prompt_tokens` and `run.usage.completion_tokens`
- raw: `thread_id`, `run_ids`, status

Data that leaves your environment: test questions are sent to your own
Foundry project endpoint only. The agent's tools (search, Bing, and so on) run
inside your Azure subscription as you configured them. Threads are deleted
after each test case unless you set `delete_threads: false`.

## AWS Bedrock Agents (`bedrock_agent`)

**Status: mock-tested (implemented against the documented API; not yet verified against a live service)**

API: `bedrock-agent-runtime` `InvokeAgent` (boto3 `invoke_agent`) with
`enableTrace=True`. The adapter reads the response event stream.
`health_check` calls `bedrock-agent` `GetAgentAlias`.

```yaml
system:
  type: bedrock_agent
  agent_id: ABCDEFGHIJ
  agent_alias_id: TSTALIASID
  region: us-east-1
  session_attributes: { tenant: northwind }
  timeout: 120
```

Options: `agent_id`, `agent_alias_id`, `region`, `profile`,
`session_attributes`, `prompt_session_attributes`, `memory_id`, `enable_trace`
(default `true`), `timeout`, `max_return_control_rounds`.

Credentials: the standard AWS credential chain, or a named `profile`. The
chain covers environment variables, SSO, an IAM role, and an instance or task
role. The caller needs the `bedrock:InvokeAgent` permission, and
`bedrock:GetAgentAlias` for health checks.

What it captures:
- answer: the `chunk` bytes joined together; if there are none, the trace's `finalResponse`
- trace: action-group calls (tool = the function name or API path; args = the
  parameters plus `_action_group`), `knowledge_base` lookups (args `{query,
  knowledge_base_id}`), `code_interpreter` and `agent:<collaborator>` calls,
  each with its observation as the result
- retrieved: knowledge-base references and chunk citations. The id is the
  source URI (S3, web, Confluence, SharePoint, Salesforce, Kendra or custom),
  and `metadata.cited` marks references cited in the answer
- tokens: the sum of `modelInvocationOutput.metadata.usage` over the
  pre-processing, orchestration and post-processing traces
- raw: `session_id`, guardrail action, failure reason

Each test case gets a new `sessionId`. A multi-turn case reuses one. An error
event in the stream (such as `throttlingException`) raises an exception. A
`failureTrace` with no answer sets `error`.

Data that leaves your environment: test questions are sent to the Bedrock
Agents runtime endpoint in your AWS account and region only.

## Google Vertex AI Agent Engine and Dialogflow CX (`vertex_agent`)

**Status: mock-tested (implemented against the documented API; not yet verified against a live service)**

This adapter has two modes, set with the `mode` option. Both call REST APIs
through `requests`.

**`mode: agent_engine`** uses Vertex AI Agent Engine (reasoning engines),
`aiplatform.googleapis.com/v1`.
- `engine_api: adk` (the default, for the ADK template) calls
  `:streamQuery?alt=sse` with `class_method: stream_query` and input
  `{user_id, message, session_id?}`. Multi-turn cases first create a session
  with `:query` and `class_method: create_session`.
- `engine_api: query` (for LangChain, LangGraph or custom templates) calls
  `:query` with `{"input": {<input_key>: question}}`. There is no native
  session here, so multi-turn cases replay the conversation as a transcript.

```yaml
system:
  type: vertex_agent
  mode: agent_engine
  project: my-gcp-project
  location: us-central1
  reasoning_engine_id: "1234567890123456789"
  engine_api: adk
  user_id: evalkit
```

**`mode: dialogflow_cx`** covers Dialogflow CX and Vertex AI Agent Builder
conversational agents (playbooks). It calls
`dialogflow.googleapis.com/v3 .../agents/{agent}/sessions/{session}:detectIntent`
with `populateDataStoreConnectionSignals: true`.

```yaml
system:
  type: vertex_agent
  mode: dialogflow_cx
  project: my-gcp-project
  location: global
  agent_id: 0b1c2d3e-aaaa-bbbb-cccc-1234567890ab
  language_code: en
  environment: null
```

Options: `mode`, `project`, `location`, `reasoning_engine_id` (a number, or the
full resource name), `engine_api`, `query_method`, `create_session_method`,
`input_key`, `extra_input`, `user_id`, `agent_id`, `environment`,
`language_code`, `query_params`, `api_version`, `timeout`, `auth`.

Credentials: Google Application Default Credentials by default (a service
account key, workload identity, or `gcloud auth application-default login`).
You can also give any evalkit `auth:` block, for example
`{type: gcp_service_account}` or `{type: bearer, token: ...}`. From Python you
can pass an already-authorized session instead (for example
`google.auth.transport.requests.AuthorizedSession`). The caller needs
`aiplatform.reasoningEngines.query` (Vertex AI User) or
`dialogflow.sessions.detectIntent` (Dialogflow API Client).

What it captures:
- `agent_engine` with `adk`: the final model text (text before the last tool
  result is dropped); `function_call` / `function_response` pairs as the trace;
  `usage_metadata` tokens; `grounding_metadata` chunks as retrieved documents
- `agent_engine` with `query`: the `output` text; LangChain
  `intermediate_steps` as the trace
- `dialogflow_cx`: `responseMessages` text as the answer; tool uses, playbook
  invocations and flow invocations from `generativeInfo.actionTracingInfo` as
  the trace; data store `searchSnippets` and `citedSnippets` as retrieved
  documents; intent, match type and page in `raw`. Dialogflow does not report
  token counts.

Data that leaves your environment: test questions are sent to the Vertex AI or
Dialogflow endpoint of your GCP project and location only.

## Microsoft Copilot Studio (`copilot_studio`)

**Status: mock-tested (implemented against the documented API; not yet verified against a live service)**

API: Bot Framework Direct Line 3.0 REST, called through `requests`. The steps:
1. Get a token: `POST /tokens/generate` with the secret, or `GET token_endpoint`.
2. Start a conversation: `POST /conversations`.
3. Send the question: `POST /conversations/{id}/activities`.
4. Poll `GET /conversations/{id}/activities?watermark=...` until the bot has
   replied and no new messages have arrived for `settle_s` seconds.

```yaml
system:
  type: copilot_studio
  direct_line_secret: ${secret:copilot-directline-secret}
  # or: token_endpoint: https://<env>.environment.api.powerplatform.com/powervirtualagents/botsbyschema/<schema>/directline/token?api-version=2022-03-01-preview
  domain: https://directline.botframework.com/v3/directline
  timeout: 60
  settle_s: 3.0
```

Options: `direct_line_secret`, `token_endpoint`, `token` (a static Direct Line
token), `domain` (use a regional host such as
`https://europe.directline.botframework.com/v3/directline` when your
environment needs one), `user_id`, `user_name`, `locale`, `timeout`,
`poll_interval`, `settle_s`, `send_start_event` (sends the `startConversation`
event first; the greeting is discarded), `include_card_text`.

Credentials: the Direct Line secret is under Copilot Studio → Settings →
Security → Web channel security. The token endpoint is under Channels → Mobile
app. Agents that need end-user sign-in are not supported.

What it captures:
- answer: the text of the bot's message activities that reply to the question,
  plus TextBlock and FactSet text from Adaptive Cards
- retrieved: schema.org `Claim` citations from generative answers (`entities[].citation`)
- trace: generative-orchestration `DynamicPlanStepTriggered` and
  `DynamicPlanStepFinished` events, when the bot sends them
- raw: `conversation_id`, the number of bot activities

Direct Line does not report token usage.

Data that leaves your environment: test questions are sent to the Direct Line
service (`directline.botframework.com` or the regional host you set). From
there they reach your Copilot Studio agent in your Power Platform environment.
