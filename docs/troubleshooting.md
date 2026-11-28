# Troubleshooting

| Message | Meaning and fix |
|---|---|
| `plugin 'x:y' needs the package 'z' ... pip install 'evalkit[extra]'` | The integration's SDK isn't installed. Install the extra (or bake it into the image with `--build-arg EXTRAS=...`). |
| `unknown system plugin 'x'. Known: [...]` | Typo in `type`, or a third-party plugin package isn't installed in this environment. |
| `invalid config ...: unknown top-level key` / `did you mean` | A typo in the YAML; the suggestion names the intended key. |
| `config for system:x is missing required option(s)` / `unknown option(s)` | The plugin's options don't match; see its doc in `docs/integrations/`. |
| `environment variable 'X' is not set` / `Key Vault ...: cannot read` | A `${secret:...}` reference couldn't be resolved: check the name, the provider and the identity's read permission. |
| `data egress blocked by privacy.allowed_hosts` | A destination isn't on the allowlist. Add it only if sending evaluation data there is approved. |
| `run incomplete: token budget reached` (gate failure) | The budget stopped the run. Raise `budget` or continue later: `evalkit resume RUN_DIR`. |
| Many `error` results with 429 / rate limit | Lower `run.concurrency`, set `run.rate_limit`, add `run.retries`; then `evalkit retry-errors RUN_DIR`. The report shows task success with errors excluded. |
| A run was killed / the machine rebooted | `evalkit resume RUN_DIR`: finished cases are kept, the rest run. |
| `no recorded trace for this input` (otel_replay) | The dataset question doesn't match any trace's input. Check content capture is on in the instrumentation, or import traces with `evalkit otel import`. |
| Trace warnings: `no input found` / `no output found` | The instrumentation isn't recording message content, or uses attribute names evalkit doesn't know; open an issue with a sanitized span. |
| `authentication required` (401) | Missing/expired token or API key. Browser: sign in again. CI: check the `x-api-key` secret. |
| `... lacks 'run'` (403) | The principal's role doesn't allow it. Map an IdP group to `runner`/`admin` in `auth.oidc.role_map`. |
| `invalid token: InvalidAudienceError` | The access token was issued for another API. Request it with your evalkit API's scope / audience. |
| `auth.mode none is for local development only` | Configure `oidc` or `api_key` before listening on a non-local address. |
| Empty answers from a local "thinking" model | Increase `max_tokens`; the model may spend its budget on hidden reasoning. |
| `rescore`/`compare` crash on Windows with UnicodeDecodeError | Fixed in 0.2.0 (all files are read as UTF-8). Upgrade. |
| Identity tests report `unknown` ACLs | The ACL source couldn't map documents to items: add provider references to doc metadata or a `doc_map`. |

Debug logging: `evalkit -v run CONFIG` (secrets and PII are still masked).
Health of every configured component: `evalkit plugins health CONFIG`.
