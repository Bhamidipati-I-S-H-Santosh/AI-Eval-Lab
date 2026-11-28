# evalkit documentation

evalkit is a vendor-neutral platform for evaluating AI agents, RAG systems and
retrievers. One engine runs everywhere: on a laptop, in CI, in a container, on
Kubernetes, against agents on Azure, AWS, Google Cloud, local models or any
framework that emits OpenTelemetry traces.

| Start here | |
|---|---|
| [Installation](installation.md) | pip extras, Docker, first run |
| [Configuration](configuration.md) | every config block, with examples |
| [What is verified](verification-status.md) | which integrations were run live vs. tested with fakes |

| Integrations | |
|---|---|
| [Agent systems](integrations/agent-systems.md) | OpenAI Responses/Assistants, Azure AI Foundry, Bedrock Agents, Vertex AI, Copilot Studio |
| [Frameworks and MCP](integrations/frameworks.md) | LangChain, LlamaIndex, CrewAI, Semantic Kernel, MCP tool servers |
| [Retrievers](integrations/retrievers.md) | Qdrant, pgvector, Pinecone, Weaviate, Elasticsearch/OpenSearch, Azure AI Search, Vertex AI Search |
| [Models, storage, secrets, auth](integrations/platform.md) | judges and providers, where results live, secrets managers, outbound auth |
| [OpenTelemetry](opentelemetry.md) | evaluate any agent from its traces; the OTLP receiver |
| [Writing a plugin](plugins.md) | add an integration without touching the engine |

| Operating it | |
|---|---|
| [Deployment](deployment.md) | server, dashboard, SSO, Docker, Helm, CI/CD templates, distributed runs |
| [Security and privacy](security.md) | data flows, egress allowlist, PII redaction, RBAC, audit log, retention |
| [Identity-aware testing](identity-testing.md) | catch agents that leak documents across permission boundaries |
| [Production monitoring](monitoring.md) | score live traffic, version tracking, regression alerts |
| [Troubleshooting](troubleshooting.md) | common errors and what they mean |
| [Licensing](licensing.md) | the current license and an evaluation of open-core options |

The evaluation method itself (metrics, abstention scoring, injection testing,
reliable numbers) is described in [EVALKIT.md](../EVALKIT.md).
