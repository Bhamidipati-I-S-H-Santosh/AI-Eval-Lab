# Models, storage, secrets and auth

Status per plugin: see [verification-status.md](../verification-status.md).

## Model providers (judges and test generation)

The judge sends grading prompts (question, reference answer, the system's
answer and retrieved context) to the provider you configure. Use a model inside
your own tenant for sensitive data.

| `provider` | Options | Credentials |
|---|---|---|
| `openai_compatible` (default) | `base_url`, `api_key`, `auth`, `timeout` | key or any [auth](#outbound-auth) block. Works for OpenAI, Groq, Together, vLLM, LM Studio, LiteLLM proxy, Ollama `/v1` |
| `azure_openai` | `azure_endpoint`, `api_version`, `api_key` or `auth: entra_id` | keyless via managed / workload identity |
| `anthropic` | `api_key`, `base_url`, `default_max_tokens` | `ANTHROPIC_API_KEY` if unset |
| `bedrock` | `region`, `profile` | AWS credential chain (IAM role, SSO, IRSA). Uses the Converse API |
| `vertex` | `project`, `location` | Application Default Credentials / workload identity |
| `ollama` | `base_url` (default `http://localhost:11434`), `think` | none; nothing leaves the machine |

```yaml
judge:
  provider: bedrock
  model: anthropic.claude-3-5-sonnet-20240620-v1:0
  region: eu-central-1
```

Judge results are cached on disk (`cache: plain`), encrypted with a Fernet key
(`cache: encrypted`, `cache_key: ${secret:...}`, create a key with
`evalkit keygen`), or not at all (`cache: off`).

## Storage

Results are written one case at a time as they finish. A local copy is always
kept (reports, the dashboard); `storage` mirrors everything to shared storage,
which is what distributed workers and the server use.

| `type` | Options | Credentials |
|---|---|---|
| `local` | `root` | filesystem permissions |
| `s3` | `bucket`, `prefix`, `region`, `endpoint_url` (MinIO/R2/Ceph), `kms_key_id` | AWS chain; objects are server-side encrypted (SSE-S3, or SSE-KMS with `kms_key_id`) |
| `azure_blob` | `account_url`, `container`, `prefix` (or `connection_string`) | `DefaultAzureCredential` |
| `gcs` | `bucket`, `prefix`, `project` | ADC / workload identity |
| `postgres` | `dsn`, `schema` | in the DSN; use `${secret:...}` and `sslmode=require` |

Layout: `<run_id>/config.yaml, cases.jsonl, meta.json, results.jsonl or records/*.json, scores.jsonl, report.json, report.md, access_report.json`;
received traces under `otel/YYYY/MM/DD/`; monitoring history at `monitoring/history.jsonl`.

## Secrets

```yaml
secrets:
  providers:
    kv:    {type: azure_keyvault, vault_url: https://my-vault.vault.azure.net}
    vault: {type: vault, url: https://vault.internal:8200, mount: secret}
  default: kv
judge:
  api_key: ${secret:groq-api-key}                 # default provider
system:
  headers: {Authorization: "Bearer ${secret:vault:evalkit/bot#token}"}
```

| `type` | Name format | Auth |
|---|---|---|
| `env` | variable name | process environment / `.env` |
| `azure_keyvault` | secret name | `DefaultAzureCredential` |
| `aws_secrets_manager` | `secret-id` or `secret-id#json_key` | AWS chain |
| `gcp_secret_manager` | secret name (`version`, default `latest`) | ADC |
| `vault` | `path#key` (KV v2) | `VAULT_TOKEN` / `token`, optional `namespace` |

Resolved values are cached in memory for 5 minutes, never written to disk, and
masked (`***`) in logs and reports.

## Outbound auth

How evalkit authenticates to the system under test or to REST services. Any
`auth:` block accepts these:

| `type` | Options |
|---|---|
| `api_key` | `key`, `header` (default `Authorization`), `prefix` |
| `bearer` | `token` |
| `oauth2_client_credentials` | `token_url`, `client_id`, `client_secret`, `scope`, `audience`, `extra_params` (Entra ID, Okta, Auth0, Keycloak, Ping, Google) |
| `entra_id` | `scope` (managed identity, workload identity, `az login`) |
| `aws_sigv4` | `service`, `region` (API Gateway, Lambda URLs, OpenSearch) |
| `gcp_service_account` | `scopes`, or `target_audience` for an ID token (Cloud Run, IAP) |

```yaml
system:
  type: http
  url: https://hr-bot.internal/api/chat
  answer_field: reply.text
  auth: {type: oauth2_client_credentials, token_url: https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token,
         client_id: ${CLIENT_ID}, client_secret: ${secret:bot-client-secret}, scope: api://hr-bot/.default}
```

Tokens are cached until two minutes before they expire.
