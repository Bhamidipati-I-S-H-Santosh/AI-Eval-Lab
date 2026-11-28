# Production monitoring

Offline evals tell you a release was good; monitoring tells you it stays
good. evalkit scores a sample of real traffic from OpenTelemetry traces.

```bash
evalkit monitor evals/prod-monitor.yaml            # exit 1 if a regression alert fired
```

```yaml
name: hr-agent-prod
storage: {type: s3, bucket: evals}                 # where the server stores received traces
judge: {provider: azure_openai, model: gpt-4.1-mini, azure_endpoint: https://..., auth: entra_id}
monitoring:
  traces_prefix: otel/
  window_hours: 24
  sample_rate: 0.2              # score 20% of traces
  metrics: [error, latency_s, total_tokens, steps, faithfulness, relevance, gave_up]
  min_n: 20                     # no alerts on tiny samples
  alpha: 0.05                   # significance level
  min_delta: 0.05               # ignore significant but negligible changes
  webhook: https://hooks.slack.com/services/...     # optional; subject to privacy.allowed_hosts
```

Each window:

1. loads the traces received in the window and samples them,
2. scores them with metrics that need no reference answer: errors, latency,
   tokens, steps, faithfulness (is the answer supported by what was
   retrieved?), relevance, `gave_up` (the validated abstention judge),
3. splits results by agent version (`service.version` resource attribute):
   a model or prompt change shows up as a new version,
4. appends the window to `monitoring/history.jsonl` and compares it with the
   previous window, and the newest version with the previous one, using a
   two-proportion test for rates and a normal-approximation test for latency
   and tokens; a significant change worse than `min_delta` is an alert,
5. lists failure patterns: most frequent error types and failing tools.

Results appear in the dashboard's Monitoring tab and in
`<output_dir>/monitor-<timestamp>/report.md`. The Helm chart runs it hourly.

## Good practice

- Set `service.version` on every deployment of the agent.
- Keep a periodic offline eval with a fixed golden set alongside monitoring:
  monitoring catches drift on real traffic, the golden set catches
  regressions on known-hard cases.
- Feed interesting production traces back into the golden set:
  `evalkit otel import` then add expected answers.
