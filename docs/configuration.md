# Configuration reference

A run is described by one YAML file. Unknown keys are rejected with a
suggestion (`unknown top-level key 'systm' (did you mean 'system'?)`), and
`evalkit validate CONFIG` builds every plugin without running anything.

Any string may reference the environment or a secrets manager:
`${VAR}`, `${VAR:-default}`, `${secret:NAME}`, `${secret:provider:NAME}`.

```yaml
name: hr-agent                       # run ids are <name>-<timestamp>

system:                              # what is evaluated (any 'system' plugin)
  type: azure_foundry
  endpoint: https://my-project.services.ai.azure.com/api/projects/hr
  agent_id: asst_abc123
  version: "2025-11-14"              # optional label recorded with every run

retriever:                           # optional: score retrieval alone, generate tests, inject attacks
  type: azure_ai_search
  endpoint: https://search.search.windows.net
  index: hr-docs
  auth: {type: entra_id, scope: https://search.azure.com/.default}

dataset:
  path: datasets/hr-golden@3f2a9c1d7e10.jsonl     # versioned file (see "Datasets")
  # generate: {n: 30, unanswerable: 5}           # or generate from the retriever (review before trusting)
  tags: [hr]                                     # optional filter
  limit: 50

judge:                               # LLM judge for correctness/faithfulness/relevance/abstention
  provider: azure_openai             # openai_compatible (default) | azure_openai | anthropic | bedrock | vertex | ollama
  model: gpt-4.1-mini                # deployment / model id
  azure_endpoint: https://my-aoai.openai.azure.com
  auth: entra_id
  cache: encrypted                   # plain | encrypted | off  (cache holds prompts: customer data)
  cache_key: ${secret:evalkit-cache-key}

abstention:
  method: judge                      # judge | hybrid | regex
  judge_model: gpt-4.1-mini          # the model the abstention check was validated on
  trust_status: false                # true: the agent's own "Status:" line decides

attacks:                             # prompt-injection testing
  mode: wrapper                      # wrapper | mcp | external
  attacks: all
  cases_per_attack: 2

identity:                            # identity-aware access-control testing (docs/identity-testing.md)
  acl: {type: sharepoint, auth: {type: oauth2_client_credentials, token_url: ..., client_id: ..., client_secret: ..., scope: https://graph.microsoft.com/.default}}
  users:
    alice: {email: alice@corp.example, system: {auth: {type: bearer, token: "${secret:alice-token}"}}}
    bob:   {email: bob@corp.example,   system: {auth: {type: bearer, token: "${secret:bob-token}"}}}

run:
  repeats: 3                         # agents are not deterministic
  concurrency: 4
  judge_concurrency: 4
  rate_limit: {requests_per_minute: 60, tokens_per_minute: 200000}
  retries: {max_attempts: 3, backoff_s: 5, max_backoff_s: 60}    # transient errors only (429, 5xx, timeouts)

budget:                              # the run stops cleanly and can be resumed
  max_tokens: 5000000
  max_cost_usd: 25
cost: {input_per_1m: 0.40, output_per_1m: 1.60}

metrics: [task_success, correctness, faithfulness, abstention, injection_resisted, access_violation, latency_s]
custom_metrics: [my_pkg.metrics]     # modules that register @metric functions

gates:                               # exit code 1 if any fails
  task_success: ">=0.85"
  injection_resisted: ">=0.95"
  access_violation: "<=0"
  error: "<=0.02"

storage:                             # where runs live (local is always kept for reports)
  type: s3
  bucket: evals
  prefix: evalkit/
secrets:
  type: azure_keyvault
  vault_url: https://my-vault.vault.azure.net
privacy:
  redact_reports: true
  allowed_hosts: ["*.azure.com", "*.windows.net", "*.microsoft.com"]
monitoring:                          # used by `evalkit monitor` (docs/monitoring.md)
  window_hours: 24
  sample_rate: 0.2
output_dir: results/evalkit
```

## Test cases (JSONL)

```json
{"id": "q1", "input": "How many days of leave?", "expected_output": "24 days a year", "expected_contains": ["24"],
 "expected_sources": ["leave_policy.md"], "expected_tools": ["search_policies"], "tags": ["hr"]}
{"id": "q2", "turns": ["I'm going to London.", "What can I claim for meals?"], "expected_number": 75}
{"id": "q3", "input": "Is there a pet policy?", "unanswerable": true}
{"id": "q4", "input": "Salary bands?", "identity": "bob", "expected_sources": ["salary_bands.md"]}
```

`turns` makes a multi-turn case (systems with native sessions use them;
others get the conversation replayed as a transcript). Every field except `id`
and `input`/`turns` is optional; metrics that need a missing field are skipped,
not failed.

## Datasets and reproducibility

- Each run stores the exact cases it used (`cases.jsonl`), plus `meta.json` with
  `dataset_sha256`, `config_sha256`, the evalkit version and the system version.
- Datasets uploaded through the API are content-addressed: `name@<sha256-12>.jsonl`.
- Reports show 95% confidence intervals; `evalkit compare` uses a paired
  significance test, so noise is never reported as a change.

## Running

| Command | Purpose |
|---|---|
| `evalkit run CONFIG` | run, score, report, gate |
| `evalkit run CONFIG --resume RUN` / `evalkit resume RUN_DIR` | continue an interrupted run; finished cases are kept |
| `evalkit prepare CONFIG --run-id ID` then `evalkit run CONFIG --run-id ID --shard i/N --no-score` on N workers, then `evalkit finalize ID --config CONFIG` | distributed execution over shared storage |
| `evalkit compare A B --fail-on-regression --metrics task_success` | CI regression gate (significant drops only) |
| `evalkit rescore RUN_DIR` | new metric or judge, no system calls |
| `evalkit plugins health CONFIG` | probe every component |
