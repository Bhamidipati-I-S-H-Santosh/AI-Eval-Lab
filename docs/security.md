# Security and privacy

## Principles

1. **Self-hosted.** evalkit runs in your infrastructure. It has no telemetry,
   no phone-home, no license server. The only network traffic is to endpoints
   named in your configuration.
2. **Your models, your storage.** Judges use the model provider you configure;
   results go to the storage you configure.
3. **No keys in config files.** Secrets are `${secret:...}` references resolved
   at runtime from your secrets manager, cached in memory only, and masked in
   logs and reports. Prefer keyless identity (managed / workload identity,
   IAM roles, OIDC federation).
4. **Least privilege.** Every integration needs read-only access to what it
   evaluates; the server separates viewer, runner, ingest and admin roles.

## Where data goes

`evalkit validate CONFIG` prints every destination. Enforce it:

```yaml
privacy:
  allowed_hosts: ["*.openai.azure.com", "hr-bot.corp.example", "localhost"]
```

Startup fails if any configured destination, including implicit vendor
defaults (for example `api.openai.com` when no `base_url` is set), is not on
the list.

| Component | What it sends | To |
|---|---|---|
| System under test | test questions (and conversation turns) | the agent endpoint you configure |
| Judge | question, reference answer, the system's answer, retrieved context / tool results | the model provider you configure |
| Retriever | test questions | your search service |
| Test generation | sampled document passages | the judge's provider |
| Storage mirror | all run artifacts (answers, retrieved text, scores) | your bucket / container / database |
| Secrets provider | secret names | your secrets manager |
| ACL source (identity tests) | document ids, user emails | Microsoft Graph / Google Drive API / Confluence |
| Monitoring webhook (optional) | metric names and values only, no content | the URL you configure |
| Server OTLP receiver | inbound only | — |

## PII and sensitive content

```yaml
privacy:
  redact_reports: true        # mask PII in report.md
  redact_stored: true         # mask PII before results are written (scores then see masked text)
  detectors: [email, phone, credit_card, ssn, iban, ip, credential]
judge:
  cache: encrypted            # or off; the cache holds grading prompts
  cache_key: ${secret:evalkit-cache-key}
```

Detection is pattern-based (credit cards are Luhn-checked). It is a safety
net, not a guarantee: for regulated data, use datasets that are approved for
evaluation and keep `redact_stored` on. All evalkit log records pass through
the same redaction.

## Retention and deletion

```bash
evalkit purge CONFIG --older-than-days 90 --dry-run
evalkit purge CONFIG --older-than-days 90          # local + configured storage
```

The Helm chart schedules this daily. Deleting a single run:
`DELETE /api/v1/runs/{id}` (admin). Both are audit-logged.

## Access control and audit

- Roles: `viewer` (read), `runner` (+ start runs, upload datasets),
  `ingest` (send traces only), `admin` (+ delete, audit log).
- Every state-changing action (run started/finished, dataset uploaded, run
  deleted, traces ingested, sign-in, purge) is appended to an audit log with
  the actor, timestamp and ids, never content or secrets.
- Each entry is hash-chained to the previous one: `evalkit audit-verify`
  detects edited or deleted lines. Ship the file to your SIEM / WORM storage
  for long-term tamper resistance.

## Server hardening

- OIDC tokens are validated for signature (JWKS, with key rotation), issuer,
  audience and expiry; browser sign-in uses PKCE, state and nonce.
- Sessions are signed cookies (`session_secret` >= 32 chars from a secrets
  manager), `HttpOnly`, `SameSite=Lax`, `Secure` when `public_url` is https.
- Responses carry `Content-Security-Policy`, `X-Frame-Options: DENY`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`.
- Path parameters are validated; artifacts are served from an allowlist; the
  API cannot start arbitrary configs; uploads are size-limited and validated.
- Terminate TLS at your ingress / load balancer.

## Threats evalkit itself tests for

- Prompt injection through retrieved content and tool results
  (`attacks:` in the config; see EVALKIT.md).
- Permission leaks across identities ([identity-testing.md](identity-testing.md)).

## Reporting a vulnerability

Contact the repository owner privately; do not open a public issue.
