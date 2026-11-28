# What is verified, and how

Every integration is listed by `python -m evalkit plugins list` with one of two
statuses. This page explains them and records exactly what was exercised.

| Status | Meaning |
|---|---|
| **verified** | Exercised end to end against the real library, protocol or service: in the automated tests, or in a recorded run on real infrastructure. |
| **mock-tested** | Implemented against the vendor's documented API and covered by tests that use fake clients with the documented response shapes. **Not yet run against the live service.** Treat the first real run as an integration test. |

## Verified

| Area | What was exercised |
|---|---|
| Core engine | 170+ automated tests (173 at release of 0.2.0): scoring, gates, confidence intervals, paired significance tests, resumable runs (simulated crash mid-run), sharded distributed runs + finalize, budgets, rate limiting, retries, multi-turn. |
| Core-only install | `pip install .` into an empty virtualenv pulls 6 small dependencies and no vendor SDK; a full retrieval eval runs; asking for an uninstalled integration prints `pip install 'evalkit[x]'`. |
| `python`, `http`, `folder`, `qdrant` (local) | Used by the Northwind demo and the test suite against real code / a real local Qdrant. |
| `mcp` system | End-to-end against this repo's real MCP server over stdio (real tool schemas, calls and results), with a scripted model. |
| OpenTelemetry ingestion | Spans created with the official `opentelemetry-sdk` and encoded by the official OTLP exporter (protobuf and JSON) are parsed and evaluated; the server's `/v1/traces` accepts both encodings. |
| `openai_compatible` model / judge | Real runs against Groq (`openai/gpt-oss-120b`, `openai/gpt-oss-safeguard-20b`) recorded in `results/evalkit`. |
| `ollama` model | Live test against a local Ollama model (`EVALKIT_LIVE=1 pytest -k ollama`). Thinking models need a realistic `max_tokens`. |
| `oauth2_client_credentials` auth | Real HTTP token exchange against a local token endpoint (form-encoded client credentials, caching, 401 handling). |
| Server API + RBAC + OIDC | Starlette app tested in-process: API keys, OIDC bearer tokens signed with a real RSA key (signature, issuer, audience, expiry, role mapping), PKCE login state/nonce checks, security headers, audit trail. |
| Identity-aware testing (static ACLs) | End-to-end runs catch a leaking agent per identity via retrieved docs, tool results, citations and content. |
| Local storage, audit log, PII redaction, egress allowlist | Unit and end-to-end tests, including torn writes and audit tampering detection. |

## Mock-tested (implemented; needs a first live run)

| Category | Plugins |
|---|---|
| Agent systems | `openai_responses`, `openai_assistants`, `azure_foundry` (classic threads/runs Agent Service; Foundry "v2" Responses-based agents are not covered yet), `bedrock_agent`, `vertex_agent` (Agent Engine, Dialogflow CX), `copilot_studio` (Direct Line 3.0) |
| Frameworks | `langchain`, `llamaindex`, `crewai`, `semantic_kernel` |
| Model providers | `azure_openai`, `anthropic`, `bedrock`, `vertex` |
| Retrievers | `pgvector`, `pinecone`, `weaviate`, `elasticsearch`, `opensearch`, `azure_ai_search`, `vertex_search` |
| Storage | `s3`, `azure_blob`, `gcs`, `postgres` |
| Secrets | `azure_keyvault`, `aws_secrets_manager`, `gcp_secret_manager`, `vault` |
| Auth | `entra_id`, `aws_sigv4`, `gcp_service_account` |
| ACL sources | `sharepoint` (Microsoft Graph), `google_drive`, `confluence` |

Each integration's doc (`docs/integrations/`) lists the specific API details
that were inferred from documentation rather than observed, so a first live run
knows what to check.

## Deployment artifacts

| Artifact | Status |
|---|---|
| Python package (`pyproject.toml`) | Verified locally (clean-venv install). |
| Dockerfile | Written, not built on the development machine (no Docker there). The repo CI job `container` builds it, starts it, checks `/healthz` and that it runs as UID 10001. |
| Helm chart | Written, not rendered locally (no Helm there). The repo CI job `helm` runs `helm lint` and `helm template` with every optional resource enabled. |
| CI templates (GitHub Actions, Azure DevOps, GitLab, Jenkins) | YAML syntax checked; the commands they run are the verified CLI. Not executed in those CI systems. |
| docker-compose + OpenTelemetry Collector config | YAML syntax checked; not started. |

## To promote a plugin to verified

1. Run it once against the real service with a small dataset
   (`python -m evalkit smoke CONFIG --n 3`, then `run`).
2. Check the fields listed under "uncertain" in its doc (trace, citations, tokens).
3. Add a `live` test (skipped unless `EVALKIT_LIVE=1`) and change its status in
   `evalkit/plugins.py`.
