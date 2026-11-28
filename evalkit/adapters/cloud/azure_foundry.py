"""Azure AI Foundry Agent Service (azure-ai-projects >= 1.0 / azure-ai-agents >= 1.0).

config:
  type: azure_foundry
  endpoint: https://<resource>.services.ai.azure.com/api/projects/<project>
  agent_id: asst_abc123                   # the agent's id in the Foundry portal
  # credentials: DefaultAzureCredential by default (managed identity, workload
  # identity, `az login`, AZURE_CLIENT_ID/AZURE_TENANT_ID/AZURE_CLIENT_SECRET env vars)
  managed_identity_client_id: null        # user-assigned managed identity
  tenant_id: null                         # these three together -> ClientSecretCredential
  client_id: null
  client_secret: ${secret:foundry-sp-secret}
  timeout: 180                            # max seconds per run
  poll_interval: 1.0
  delete_threads: true
  auto_approve_tools: false               # approve MCP tool calls that need approval

Flow per test case: threads.create -> messages.create -> runs.create (polled with
a bounded timeout; set use_create_and_process: true to use the SDK's unbounded
runs.create_and_process instead) -> messages.list + run_steps.list. Tool calls
(file_search, azure_ai_search, bing_grounding, sharepoint_grounding, openapi,
connected_agent, mcp, function) go to the trace; url/file citations and
file_search chunks go to `retrieved`; run.usage gives token counts.
Needs: pip install 'evalkit[azure]'.
"""
from __future__ import annotations

from typing import Any, Callable

from ...errors import ConfigError
from ...plugins import HealthStatus, timed_health
from .openai_agents import _ThreadRunAgent, _enum, _g, _items, _require

_PLUGIN = "system:azure_foundry"
_FILE_SEARCH_CONTENTS = "step_details.tool_calls[*].file_search.results[*].content"


class AzureFoundryAgent(_ThreadRunAgent):
    service = "Azure AI Foundry"

    def __init__(self, endpoint: str | None = None, agent_id: str | None = None,
                 connection_string: str | None = None, additional_instructions: str | None = None,
                 tenant_id: str | None = None, client_id: str | None = None,
                 client_secret: str | None = None, managed_identity_client_id: str | None = None,
                 timeout: float = 180, poll_interval: float = 1.0, delete_threads: bool = True,
                 use_create_and_process: bool = False, include_search_content: bool = True,
                 strip_citation_markers: bool = True, auto_approve_tools: bool = False,
                 tool_handlers: dict[str, Callable] | None = None, credential=None, client=None, **_):
        if not agent_id:
            raise ConfigError("azure_foundry needs 'agent_id' (shown on the agent's page in the Foundry portal)")
        if client is None:
            if not endpoint and not connection_string:
                raise ConfigError("azure_foundry needs 'endpoint' "
                                  "(https://<resource>.services.ai.azure.com/api/projects/<project>)")
            projects = _require("azure.ai.projects", _PLUGIN, "azure")
            if credential is None:
                credential = self._credential(tenant_id, client_id, client_secret, managed_identity_client_id)
            if endpoint:
                client = projects.AIProjectClient(endpoint=endpoint, credential=credential)
            else:  # pre-GA hub-based projects (azure-ai-projects < 1.0.0b10)
                client = projects.AIProjectClient.from_connection_string(conn_str=connection_string,
                                                                         credential=credential)
        self.client = client
        # AIProjectClient exposes the agents client as .agents; an AgentsClient may be injected directly.
        self.agents = getattr(client, "agents", None) or client
        self.agent_id, self.endpoint = agent_id, endpoint
        self.additional_instructions = additional_instructions
        self.timeout, self.poll_interval, self.delete_threads = float(timeout), float(poll_interval), delete_threads
        self.use_create_and_process = use_create_and_process
        self.include_search_content, self.strip_citation_markers = include_search_content, strip_citation_markers
        self.auto_approve_tools = auto_approve_tools
        self.tool_handlers = dict(tool_handlers or {})

    @staticmethod
    def _credential(tenant_id, client_id, client_secret, mi_client_id):
        identity = _require("azure.identity", _PLUGIN, "azure")
        if client_secret or (tenant_id and client_id and not mi_client_id):
            if not (tenant_id and client_id and client_secret):
                raise ConfigError("azure_foundry service-principal auth needs tenant_id, client_id and "
                                  "client_secret together (or omit all three to use DefaultAzureCredential)")
            return identity.ClientSecretCredential(tenant_id=tenant_id, client_id=client_id,
                                                   client_secret=client_secret)
        kw = {"managed_identity_client_id": mi_client_id} if mi_client_id else {}
        return identity.DefaultAzureCredential(**kw)

    # --- SDK compatibility: azure-ai-agents 1.x sub-clients, falling back to the
    # flat method names of the azure-ai-projects 1.0.0b* previews. ---
    def _op(self, new: str, old: str) -> Callable:
        obj: Any = self.agents
        try:
            for part in new.split("."):
                obj = getattr(obj, part)
            if callable(obj):
                return obj
        except AttributeError:
            pass
        fn = getattr(self.agents, old, None)
        if not callable(fn):
            raise AttributeError(f"the Azure agents client has neither '{new}' nor '{old}'; "
                                 f"upgrade azure-ai-projects / azure-ai-agents to >= 1.0")
        return fn

    def _op_create_thread(self):
        return self._op("threads.create", "create_thread")()

    def _op_post_message(self, thread_id, text):
        return self._op("messages.create", "create_message")(thread_id=thread_id, role="user", content=text)

    def _run_kwargs(self, thread_id: str) -> dict:
        kw: dict[str, Any] = {"thread_id": thread_id, "agent_id": self.agent_id}
        if self.additional_instructions:
            kw["additional_instructions"] = self.additional_instructions
        return kw

    def _op_create_run(self, thread_id):
        return self._op("runs.create", "create_run")(**self._run_kwargs(thread_id))

    def _op_get_run(self, thread_id, run_id):
        return self._op("runs.get", "get_run")(thread_id=thread_id, run_id=run_id)

    def _op_cancel_run(self, thread_id, run_id):
        return self._op("runs.cancel", "cancel_run")(thread_id=thread_id, run_id=run_id)

    def _op_submit(self, thread_id, run_id, **kw):
        return self._op("runs.submit_tool_outputs", "submit_tool_outputs_to_run")(
            thread_id=thread_id, run_id=run_id, **kw)

    def _op_list_messages(self, thread_id, run_id):
        # filtered to this run client-side (ThreadMessage.run_id) by the engine
        return _items(self._op("messages.list", "list_messages")(thread_id=thread_id, order="asc"))

    def _op_list_steps(self, thread_id, run_id):
        fn = self._op("run_steps.list", "list_run_steps")
        if self.include_search_content:
            try:
                return _items(fn(thread_id=thread_id, run_id=run_id, include=[_FILE_SEARCH_CONTENTS]))
            except TypeError:  # older SDK without `include`
                pass
        return _items(fn(thread_id=thread_id, run_id=run_id))

    def _op_delete_thread(self, thread_id):
        return self._op("threads.delete", "delete_thread")(thread_id=thread_id)

    def _process_run(self, thread_id):
        if not self.use_create_and_process:
            return super()._process_run(thread_id)
        run = self._op("runs.create_and_process", "create_and_process_run")(**self._run_kwargs(thread_id))
        status = _enum(_g(run, "status", ""))
        if status == "requires_action":    # SDK returns it when no toolset can serve the call
            return self._wait(thread_id, run)
        return run, None

    def health_check(self) -> HealthStatus:
        def probe():
            a = self.agents.get_agent(self.agent_id)
            return f"agent '{_g(a, 'name', self.agent_id)}' ({_g(a, 'model', '?')}) reachable"
        return timed_health(probe)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
