"""The plugin system.

Every integration (agent system, retriever, model provider, storage, secrets,
auth) is a plugin: a class registered under a (category, name) pair. The core
engine only ever talks to the category's interface, so adding an integration
never touches the engine.

Registration, three ways:
  1. Built-ins below are registered by import path, so a vendor SDK is only
     imported when a config actually uses that plugin.
  2. Third-party packages declare entry points (no evalkit change needed):
        [project.entry-points."evalkit.system"]
        my_agent = "my_pkg.adapter:MyAgent"
  3. In code: @register("system", "my_agent").

Config validation: a plugin may define a pydantic model `Config`; otherwise
its __init__ signature is checked (missing required args, unknown keys).
"""
from __future__ import annotations

import importlib
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import PluginConfigError, PluginDependencyError, PluginNotFoundError

log = logging.getLogger("evalkit.plugins")

CATEGORIES = ("system", "retriever", "model", "storage", "secrets", "auth", "identity")

# Verification status shown by `evalkit plugins list`. Be honest here:
#   verified     - exercised end to end against the real service/library in this repo's tests or docs
#   mock-tested  - implemented against the vendor's documented API, tested with a fake client only
VERIFIED, MOCK_TESTED = "verified", "mock-tested"


@dataclass
class HealthStatus:
    ok: bool
    detail: str = ""
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "detail": self.detail, "latency_ms": self.latency_ms}


class Plugin:
    """Lifecycle every plugin shares. Override what applies."""

    #: optional pydantic model used to validate the config block
    Config: Any = None

    def health_check(self) -> HealthStatus:
        """Cheap call that proves the plugin can reach its backend."""
        return HealthStatus(True, "no health check implemented")

    def close(self) -> None:
        """Release connections. Called once when the run ends."""


@dataclass
class PluginSpec:
    category: str
    name: str
    target: str | type
    extra: str | None = None           # pip extra that provides its dependencies
    description: str = ""
    status: str = VERIFIED
    source: str = "builtin"            # builtin | entry-point | code

    def load(self) -> type:
        if not isinstance(self.target, str):      # a class, or any factory callable
            return self.target
        module, _, attr = self.target.partition(":")
        try:
            obj = importlib.import_module(module)
        except ModuleNotFoundError as e:
            # Missing vendor SDK -> actionable message; missing evalkit module -> real bug.
            if e.name and not e.name.startswith(("evalkit", "examples", "src", "tests")):
                raise PluginDependencyError(f"{self.category}:{self.name}", e.name, self.extra) from e
            raise
        for part in attr.split("."):
            obj = getattr(obj, part)
        self.target = obj
        return obj


@dataclass
class Registry:
    specs: dict[tuple[str, str], PluginSpec] = field(default_factory=dict)
    _discovered: bool = False

    def register(self, category: str, name: str, target: str | type, *, extra: str | None = None,
                 description: str = "", status: str = VERIFIED, source: str = "code") -> None:
        if category not in CATEGORIES:
            raise ValueError(f"unknown plugin category '{category}'; known: {CATEGORIES}")
        self.specs[(category, name)] = PluginSpec(category, name, target, extra, description, status, source)

    def discover(self) -> None:
        """Load third-party plugins from installed packages' entry points."""
        if self._discovered:
            return
        self._discovered = True
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover
            return
        for category in CATEGORIES:
            for ep in entry_points(group=f"evalkit.{category}"):
                if (category, ep.name) not in self.specs:
                    self.register(category, ep.name, ep.value, source="entry-point",
                                  description=f"from {getattr(ep, 'dist', None) and ep.dist.name or 'entry point'}")

    def get(self, category: str, name: str) -> type:
        self.discover()
        spec = self.specs.get((category, name))
        if spec is None:
            known = sorted(n for c, n in self.specs if c == category)
            raise PluginNotFoundError(f"unknown {category} plugin '{name}'. Known: {known}, or use "
                                      f"type: custom with class: module:Class")
        return spec.load()

    def list(self, category: str | None = None) -> list[PluginSpec]:
        self.discover()
        return sorted((s for s in self.specs.values() if category in (None, s.category)),
                      key=lambda s: (s.category, s.name))

    def build(self, category: str, cfg: dict, **extra) -> Any:
        """cfg = {"type": name, ...options}. 'type: custom' + 'class: mod:Cls' loads any class."""
        from .secrets import resolve_config
        cfg = resolve_config(dict(cfg))
        if category == "system":
            cfg.pop("version", None)      # a label recorded in run metadata, not a plugin option
        name = cfg.pop("type", None)
        if not name:
            raise PluginConfigError(f"{category} config needs a 'type'")
        if name == "custom":
            if "class" not in cfg:
                raise PluginConfigError(f"{category} type 'custom' needs 'class: module:Class'")
            cls = PluginSpec(category, "custom", cfg.pop("class")).load()
        else:
            cls = self.get(category, name)
        options = validate_options(cls, {**cfg, **extra}, f"{category}:{name}")
        try:
            return cls(**options)
        except ModuleNotFoundError as e:
            if e.name and not e.name.startswith(("evalkit", "examples", "src", "tests")):
                spec = self.specs.get((category, name))
                raise PluginDependencyError(f"{category}:{name}", e.name, spec.extra if spec else None) from e
            raise


def validate_options(cls: type, options: dict, label: str) -> dict:
    """Validate a config block against the plugin's pydantic Config or its __init__."""
    model = getattr(cls, "Config", None)
    if model is not None and hasattr(model, "model_validate"):
        try:
            validated = model.model_validate(options)
        except Exception as e:  # pydantic.ValidationError
            raise PluginConfigError(f"invalid config for {label}: {e}") from e
        return {**options, **validated.model_dump(exclude_unset=True)}
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return options
    params = [p for p in sig.parameters.values() if p.name != "self"]
    accepts_any = any(p.kind is p.VAR_KEYWORD for p in params)
    names = {p.name for p in params if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    missing = [p.name for p in params if p.default is p.empty
               and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and p.name not in options]
    if missing:
        raise PluginConfigError(f"config for {label} is missing required option(s): {missing}")
    unknown = sorted(set(options) - names)
    if unknown and not accepts_any:
        raise PluginConfigError(f"config for {label} has unknown option(s) {unknown}; "
                                f"allowed: {sorted(names)}")
    if unknown:
        log.debug("%s ignores unknown options %s", label, unknown)
    return options


def timed_health(fn) -> HealthStatus:
    """Run a health probe, catching errors into a HealthStatus."""
    t0 = time.perf_counter()
    try:
        detail = fn() or "ok"
        return HealthStatus(True, str(detail), round((time.perf_counter() - t0) * 1000, 1))
    except Exception as e:
        return HealthStatus(False, f"{type(e).__name__}: {e}", round((time.perf_counter() - t0) * 1000, 1))


REGISTRY = Registry()


def register(category: str, name: str, **kw):
    """Decorator: @register("system", "my_agent")."""
    def deco(cls):
        REGISTRY.register(category, name, cls, **kw)
        return cls
    return deco


# ---------- built-ins (import paths: nothing vendor-specific is imported here) ----------
_B = [
    # systems (agents under test)
    ("system", "python", "evalkit.adapters.python_agent:PythonAgentAdapter", None, "any Python callable", VERIFIED),
    ("system", "http", "evalkit.adapters.http_agent:HttpAgentAdapter", None, "any JSON HTTP API", VERIFIED),
    ("system", "mcp", "evalkit.adapters.mcp_agent:McpToolAgent", "mcp", "agent loop over an MCP tool server", VERIFIED),
    ("system", "otel_replay", "evalkit.otel.adapter:OtelReplayAgent", None, "evaluate recorded OpenTelemetry traces", VERIFIED),
    ("system", "openai_responses", "evalkit.adapters.cloud.openai_agents:OpenAIResponsesAgent", "openai", "OpenAI Responses API", MOCK_TESTED),
    ("system", "openai_assistants", "evalkit.adapters.cloud.openai_agents:OpenAIAssistantsAgent", "openai", "OpenAI Assistants API (threads/runs)", MOCK_TESTED),
    ("system", "azure_foundry", "evalkit.adapters.cloud.azure_foundry:AzureFoundryAgent", "azure", "Azure AI Foundry Agent Service", MOCK_TESTED),
    ("system", "bedrock_agent", "evalkit.adapters.cloud.bedrock_agent:BedrockAgent", "aws", "AWS Bedrock Agents", MOCK_TESTED),
    ("system", "vertex_agent", "evalkit.adapters.cloud.vertex_agent:VertexAgent", "gcp", "Google Vertex AI Agent Engine / Agent Builder", MOCK_TESTED),
    ("system", "copilot_studio", "evalkit.adapters.cloud.copilot_studio:CopilotStudioAgent", None, "Microsoft Copilot Studio (Direct Line)", MOCK_TESTED),
    ("system", "langchain", "evalkit.adapters.frameworks:LangChainAgent", "langchain", "LangChain / LangGraph runnable", MOCK_TESTED),
    ("system", "llamaindex", "evalkit.adapters.frameworks:LlamaIndexAgent", "llamaindex", "LlamaIndex query engine / agent", MOCK_TESTED),
    ("system", "crewai", "evalkit.adapters.frameworks:CrewAIAgent", "crewai", "CrewAI crew", MOCK_TESTED),
    ("system", "semantic_kernel", "evalkit.adapters.frameworks:SemanticKernelAgent", "semantic-kernel", "Semantic Kernel agent/function", MOCK_TESTED),
    # retrievers
    ("retriever", "folder", "evalkit.adapters.folder:FolderRetrieverAdapter", None, "BM25 over local .md/.txt", VERIFIED),
    ("retriever", "qdrant", "evalkit.adapters.qdrant:QdrantRetrieverAdapter", "qdrant", "Qdrant", VERIFIED),
    ("retriever", "pgvector", "evalkit.adapters.retrievers.pgvector:PgVectorRetriever", "pgvector", "PostgreSQL + pgvector", MOCK_TESTED),
    ("retriever", "pinecone", "evalkit.adapters.retrievers.pinecone:PineconeRetriever", "pinecone", "Pinecone", MOCK_TESTED),
    ("retriever", "weaviate", "evalkit.adapters.retrievers.weaviate:WeaviateRetriever", "weaviate", "Weaviate", MOCK_TESTED),
    ("retriever", "elasticsearch", "evalkit.adapters.retrievers.elastic:ElasticRetriever", None, "Elasticsearch (REST)", MOCK_TESTED),
    ("retriever", "opensearch", "evalkit.adapters.retrievers.elastic:OpenSearchRetriever", None, "OpenSearch (REST; SigV4 for AWS)", MOCK_TESTED),
    ("retriever", "azure_ai_search", "evalkit.adapters.retrievers.azure_search:AzureAISearchRetriever", None, "Azure AI Search (REST)", MOCK_TESTED),
    ("retriever", "vertex_search", "evalkit.adapters.retrievers.vertex_search:VertexSearchRetriever", "gcp", "Vertex AI Search (REST; google-auth for tokens)", MOCK_TESTED),
    # model providers (judges, test generation)
    ("model", "openai_compatible", "evalkit.providers.openai_compat:OpenAICompatibleProvider", "openai", "any OpenAI-compatible endpoint (OpenAI, Groq, vLLM, LM Studio...)", VERIFIED),
    ("model", "azure_openai", "evalkit.providers.azure_openai:AzureOpenAIProvider", "openai", "Azure OpenAI (key or Entra ID)", MOCK_TESTED),
    ("model", "anthropic", "evalkit.providers.anthropic:AnthropicProvider", "anthropic", "Anthropic Messages API", MOCK_TESTED),
    ("model", "bedrock", "evalkit.providers.bedrock:BedrockProvider", "aws", "AWS Bedrock Converse API", MOCK_TESTED),
    ("model", "vertex", "evalkit.providers.vertex:VertexProvider", "gcp", "Google Vertex AI (Gemini)", MOCK_TESTED),
    ("model", "ollama", "evalkit.providers.ollama:OllamaProvider", None, "local models via Ollama", VERIFIED),
    # storage
    ("storage", "local", "evalkit.storage.local:LocalStorage", None, "local filesystem", VERIFIED),
    ("storage", "s3", "evalkit.storage.cloud:S3Storage", "aws", "Amazon S3 (and S3-compatible)", MOCK_TESTED),
    ("storage", "azure_blob", "evalkit.storage.cloud:AzureBlobStorage", "azure", "Azure Blob Storage", MOCK_TESTED),
    ("storage", "gcs", "evalkit.storage.cloud:GCSStorage", "gcp", "Google Cloud Storage", MOCK_TESTED),
    ("storage", "postgres", "evalkit.storage.postgres:PostgresStorage", "postgres", "PostgreSQL", MOCK_TESTED),
    # secrets
    ("secrets", "env", "evalkit.secrets:EnvSecrets", None, "environment variables / .env", VERIFIED),
    ("secrets", "azure_keyvault", "evalkit.secrets:AzureKeyVaultSecrets", "azure", "Azure Key Vault", MOCK_TESTED),
    ("secrets", "aws_secrets_manager", "evalkit.secrets:AwsSecretsManager", "aws", "AWS Secrets Manager", MOCK_TESTED),
    ("secrets", "gcp_secret_manager", "evalkit.secrets:GcpSecretManager", "gcp", "GCP Secret Manager", MOCK_TESTED),
    ("secrets", "vault", "evalkit.secrets:HashiCorpVault", None, "HashiCorp Vault KV v2 (REST)", MOCK_TESTED),
    # auth (outbound credentials for calling systems under test)
    ("auth", "api_key", "evalkit.auth:ApiKeyAuth", None, "static key in a header", VERIFIED),
    ("auth", "bearer", "evalkit.auth:BearerAuth", None, "static bearer token", VERIFIED),
    ("auth", "oauth2_client_credentials", "evalkit.auth:OAuth2ClientCredentials", None, "OAuth 2.0 client credentials (Okta, Entra, Auth0, Google...)", VERIFIED),
    ("auth", "entra_id", "evalkit.auth:EntraIdAuth", "azure", "Microsoft Entra ID via azure-identity", MOCK_TESTED),
    ("auth", "aws_sigv4", "evalkit.auth:AwsSigV4Auth", "aws", "AWS IAM SigV4 request signing", MOCK_TESTED),
    ("auth", "gcp_service_account", "evalkit.auth:GcpServiceAccountAuth", "gcp", "Google service account / ADC", MOCK_TESTED),
    # identity-aware testing: who may see which document
    ("identity", "static", "evalkit.identity:StaticAclSource", None, "ACLs from a YAML/JSON file", VERIFIED),
    ("identity", "sharepoint", "evalkit.identity:SharePointAclSource", None, "SharePoint / OneDrive via Microsoft Graph", MOCK_TESTED),
    ("identity", "google_drive", "evalkit.identity:GoogleDriveAclSource", None, "Google Drive permissions API", MOCK_TESTED),
    ("identity", "confluence", "evalkit.identity:ConfluenceAclSource", None, "Confluence page restrictions API", MOCK_TESTED),
]
for _c, _n, _t, _x, _d, _s in _B:
    REGISTRY.register(_c, _n, _t, extra=_x, description=_d, status=_s, source="builtin")
