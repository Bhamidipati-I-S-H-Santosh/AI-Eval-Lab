# Writing a plugin

The engine only talks to interfaces; every integration is a plugin registered
under a `(category, name)` pair. Adding one never requires changing evalkit.

| Category | Base class | Implement |
|---|---|---|
| `system` | `evalkit.adapters.base.AgentAdapter` | `run(input) -> AgentResult`; optionally `run_conversation(turns)`, `set_retriever`, `set_identity` + `supports_identity`, `health_check`, `close` |
| `retriever` | `evalkit.adapters.base.RetrieverAdapter` | `search(query, k) -> list[RetrievedDoc]`; optionally `sample(n, seed)` |
| `model` | `evalkit.providers.ModelProvider` | `complete(messages, model, temperature, max_tokens) -> Completion` |
| `storage` | `evalkit.storage.StorageBackend` | `write_text`, `read_text`, `list`, `delete`; optionally `append_record` / `read_records` |
| `secrets` | `evalkit.secrets.SecretsProvider` | `fetch(name) -> str` |
| `auth` | `evalkit.auth.AuthProvider` | `headers()` and/or `__call__(request)`, `token()` |
| `identity` | `evalkit.identity.AclSource` | `allowed(user, doc) -> True / False / None` |

## Example

```python
# my_company_evalkit/agent.py
from evalkit import AgentAdapter, AgentResult, register
from evalkit.plugins import timed_health

@register("system", "acme_bot")
class AcmeBot(AgentAdapter):
    def __init__(self, url: str, tenant: str, client=None, **_):
        import acme_sdk                                  # import vendor SDKs inside __init__
        self.client = client or acme_sdk.Client(url, tenant)

    def run(self, input: str) -> AgentResult:
        r = self.client.ask(input)
        return AgentResult(answer=r.text, prompt_tokens=r.usage.input, completion_tokens=r.usage.output)

    def health_check(self):
        return timed_health(lambda: self.client.ping())
```

Ship it as a package with an entry point and it is discovered automatically:

```toml
[project.entry-points."evalkit.system"]
acme_bot = "my_company_evalkit.agent:AcmeBot"
```

```yaml
system: {type: acme_bot, url: https://bot.acme.example, tenant: emea}
```

Or, without packaging: `system: {type: custom, class: my_module:AcmeBot, ...}`.

## Contracts

- **Config validation.** Options arrive as keyword arguments (secrets already
  resolved). Required options are checked from your `__init__` signature;
  define a pydantic model as `Config` for type checking. Accept `**_` only if
  you deliberately ignore unknown options.
- **Lazy vendor imports.** Import SDKs inside `__init__`, never at module top
  level. A missing package then becomes `pip install 'evalkit[x]'` advice.
- **Injectable clients.** Accept `client=None` / `session=None` so tests can
  pass fakes.
- **Errors.** Raise `evalkit.errors.ConfigError` for bad options with a
  message that says what to change. For runtime failures raise or return
  `AgentResult(error=...)`; the runner records the error, retries transient
  ones, and never lets one case stop the run.
- **Status.** Declare `status="mock-tested"` (`REGISTRY.register(..., status=...)`)
  until the plugin has run against the real service.
