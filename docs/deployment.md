# Deployment

The CLI, the CI templates, the server and the scheduled jobs all run the same
engine (`evalkit.runner`). Pick the shapes you need.

## 1. CLI on a workstation or build agent

```bash
pip install "evalkit[openai]"
evalkit run evals/hr-agent.yaml
```

## 2. CI/CD quality gates

Templates in `deploy/ci/`: `github-actions.yml` (reusable workflow),
`azure-pipelines.yml`, `gitlab-ci.yml`, `Jenkinsfile`. Each:

1. installs evalkit with the extras you name,
2. `evalkit validate` (prints where data will be sent),
3. `evalkit run` (fails on the config's threshold gates),
4. on pull/merge requests, downloads the last run from the main branch and runs
   `evalkit compare BASELINE NEW --fail-on-regression --metrics ...`, which
   fails **only on statistically significant regressions**, so a noisy
   one-case dip doesn't block a merge,
5. publishes `report.md` and the run directory as an artifact.

Use the platform's workload identity / OIDC federation (GitHub `id-token`,
Azure DevOps workload identity service connection, GitLab `id_tokens`) so no
cloud keys are stored in CI.

## 3. Server: API, OTLP receiver, dashboard

```bash
pip install "evalkit[server,openai]"
evalkit serve --config deploy/examples/server.yaml --host 0.0.0.0 --port 8080
```

| Endpoint | Role | Purpose |
|---|---|---|
| `GET /healthz`, `/readyz` | none | liveness / readiness (storage reachable) |
| `GET /` | signed in | dashboard: runs, reports, jobs, monitoring, plugins, audit |
| `GET /api/v1/runs`, `/runs/{id}`, `/runs/{id}/artifacts/{report.md,report.json,access_report.json,scores.jsonl,meta.json}` | viewer | results |
| `POST /api/v1/runs {"config": "<name>", "limit": n}` | runner | start a run of an admin-provided config |
| `POST /api/v1/runs/{id}/resume` | runner | continue an interrupted run |
| `DELETE /api/v1/runs/{id}` | admin | delete a run |
| `GET /api/v1/jobs`, `/jobs/{id}` | viewer | background run status |
| `GET /api/v1/datasets`, `PUT /api/v1/datasets/{name}` | viewer / runner | list; upload (validated, stored as `name@<hash>.jsonl`) |
| `POST /v1/traces` | ingest | OTLP/HTTP receiver (JSON or protobuf) |
| `GET /api/v1/monitoring` | viewer | monitoring windows and alerts |
| `GET /api/v1/audit` | admin | audit log and hash-chain check |
| `GET /api/v1/plugins`, `/configs`, `/me` | viewer | catalogue, startable configs, current identity |

**The API never accepts a config body.** Configs can name Python classes, so
accepting them over HTTP would allow code execution. Runs can only start
configs an administrator placed in `configs_dir`.

### Single sign-on

`auth.mode: oidc` works with any OpenID Connect provider:

| IdP | `issuer` | Roles claim |
|---|---|---|
| Microsoft Entra ID | `https://login.microsoftonline.com/<tenant>/v2.0` | `roles` (app roles) |
| Okta | `https://<org>.okta.com/oauth2/default` | `groups` |
| Google | `https://accounts.google.com` | none: use `default_role` + API keys for elevated roles |
| Keycloak / Auth0 / Ping | the realm / tenant issuer URL | configurable |

The dashboard uses the authorization-code flow with PKCE, state and nonce
checks; API clients send `Authorization: Bearer <access token>` (validated
against the IdP's JWKS: signature, issuer, audience, expiry). CI jobs and
OTLP exporters use API keys (`x-api-key`), stored only as SHA-256 hashes in
memory. `auth.mode: none` is refused on anything but localhost.

Register a redirect URI of `<public_url>/auth/callback` with the IdP.

## 4. Docker / docker-compose

```bash
docker build -t evalkit:0.2.0 --build-arg EXTRAS="server,openai,aws" .
docker compose -f deploy/docker-compose.yml up -d     # evalkit + OpenTelemetry Collector
```

## 5. Kubernetes (Helm)

```bash
kubectl create secret generic evalkit-secrets --from-env-file=evalkit.env   # or External Secrets / CSI driver
helm install evalkit deploy/helm/evalkit -f my-values.yaml
```

The chart: non-root, read-only root filesystem, all capabilities dropped,
seccomp `RuntimeDefault`, a NetworkPolicy (optional egress restriction),
workload-identity service account annotations (IRSA, Azure WI, GKE WI),
config checksums for rollouts, a PVC for local storage, and CronJobs for
`evalkit monitor` and retention (`evalkit purge`).

## 6. Distributed runs

For large datasets, run shards in parallel against shared storage:

```bash
evalkit prepare evals/big.yaml --run-id nightly-42                 # freezes the cases (incl. attack tokens)
evalkit run evals/big.yaml --run-id nightly-42 --shard 0/8 --no-score   # on 8 workers (k8s Job, CI matrix)
...
evalkit finalize nightly-42 --config evals/big.yaml                # merge, score, report, gate
```

Every case is assigned to exactly one shard by a stable hash; a failed worker
is simply re-run (finished cases are skipped).

## Scaling notes

- With `storage.type: local` run one API replica (the job queue is in memory).
  Use shared storage and shard workers to scale execution.
- `run.concurrency` and `run.rate_limit` bound the load on the system under
  test; `budget` bounds spend per run.
