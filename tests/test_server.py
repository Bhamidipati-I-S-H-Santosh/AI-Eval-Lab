"""REST API, OTLP receiver, monitoring, OIDC/API-key auth and RBAC."""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

pytest.importorskip("starlette")
pytest.importorskip("httpx")
from starlette.testclient import TestClient  # noqa: E402

from evalkit import audit  # noqa: E402
from evalkit.errors import AccessDeniedError, ConfigError  # noqa: E402
from evalkit.server.app import create_app  # noqa: E402
from evalkit.server.auth import OidcVerifier  # noqa: E402

VIEWER, RUNNER, ADMIN, INGEST = "viewer-key-0123456789", "runner-key-0123456789", "admin-key-0123456789abc", "ingest-key-0123456789"


def echo_agent(q):
    return {"answer": f"You get 24 days [leave_policy.md] for: {q}", "prompt_tokens": 5, "completion_tokens": 5}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALKIT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    configs = tmp_path / "configs"
    configs.mkdir()
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps({"id": f"c{i}", "input": f"leave {i}?",
                                                              "expected_contains": ["24 days"]}) + "\n"
                                                  for i in range(3)), encoding="utf-8")
    (configs / "demo.yaml").write_text(yaml.safe_dump({
        "name": "demo", "system": {"type": "python", "target": "tests.test_server:echo_agent"},
        "dataset": {"path": str(tmp_path / "cases.jsonl")}, "run": {"concurrency": 1},
        "gates": {"answer_contains": ">=0.9"}}), encoding="utf-8")
    cfg = {"storage": {"type": "local", "root": str(tmp_path / "runs")}, "configs_dir": str(configs),
           "datasets_dir": str(tmp_path / "datasets"),
           "auth": {"mode": "api_key", "api_keys": {VIEWER: {"name": "viewer-bot", "roles": ["viewer"]},
                                                     RUNNER: {"name": "ci", "roles": ["runner"]},
                                                     ADMIN: {"name": "root", "roles": ["admin"]},
                                                     INGEST: {"name": "otel-collector", "roles": ["ingest"]}}}}
    app = create_app(cfg)
    return TestClient(app), app, tmp_path


def h(key):
    return {"x-api-key": key}


def _wait(client, job, key=RUNNER, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/v1/jobs/{job}", headers=h(key)).json()
        if j["status"] in ("finished", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health_is_public_everything_else_needs_auth(server):
    c, _, _ = server
    assert c.get("/healthz").json()["status"] == "ok"
    assert c.get("/readyz").status_code == 200
    assert c.get("/api/v1/runs").status_code == 401
    assert c.get("/api/v1/runs", headers=h("not-a-real-key-123456")).status_code == 401
    r = c.get("/api/v1/runs", headers=h(VIEWER))
    assert r.status_code == 200 and r.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in r.headers["content-security-policy"]


def test_rbac_and_full_run_lifecycle(server):
    c, _, tmp = server
    assert c.post("/api/v1/runs", json={"config": "demo"}, headers=h(VIEWER)).status_code == 403
    assert c.post("/api/v1/runs", json={"config": "../../etc/passwd"}, headers=h(RUNNER)).status_code == 400
    assert c.post("/api/v1/runs", json={"config": "missing"}, headers=h(RUNNER)).status_code == 404
    r = c.post("/api/v1/runs", json={"config": "demo"}, headers=h(RUNNER))
    assert r.status_code == 202
    job = _wait(c, r.json()["job"])
    assert job["status"] == "finished" and job["gates"] == "PASSED" and job["submitted_by"] == "ci (api_key)"
    rid = r.json()["run_id"]
    runs = c.get("/api/v1/runs", headers=h(VIEWER)).json()["runs"]
    assert any(x["run_id"] == rid and x["gates"] == "PASSED" for x in runs)
    detail = c.get(f"/api/v1/runs/{rid}", headers=h(VIEWER)).json()
    assert detail["report"]["overall"]["answer_contains"]["mean"] == 1.0
    assert "# Eval report" in c.get(f"/api/v1/runs/{rid}/artifacts/report.md", headers=h(VIEWER)).text
    assert c.get(f"/api/v1/runs/{rid}/artifacts/config.yaml", headers=h(VIEWER)).status_code == 404   # not exposed
    assert c.delete(f"/api/v1/runs/{rid}", headers=h(RUNNER)).status_code == 403
    assert c.delete(f"/api/v1/runs/{rid}", headers=h(ADMIN)).json()["deleted"] > 0
    events = c.get("/api/v1/audit", headers=h(ADMIN)).json()
    assert events["chain_ok"]
    names = [(e["event"], e["actor"]) for e in events["events"]]
    assert ("api.run.submitted", "ci (api_key)") in names and ("api.run.deleted", "root (api_key)") in names
    assert c.get("/api/v1/audit", headers=h(RUNNER)).status_code == 403


def test_dataset_upload_is_validated_and_versioned(server):
    c, _, tmp = server
    good = b'{"id": "a", "input": "q"}\n{"id": "b", "input": "q2"}\n'
    r = c.put("/api/v1/datasets/hr-golden", content=good, headers=h(RUNNER))
    assert r.status_code == 201 and r.json()["cases"] == 2
    again = c.put("/api/v1/datasets/hr-golden", content=good, headers=h(RUNNER)).json()
    assert again["version"] == r.json()["version"]                  # content-addressed
    assert c.put("/api/v1/datasets/hr-golden", content=b'{"id": "a", "input": "q"}\n{"id": "a", "input": "x"}\n',
                 headers=h(RUNNER)).status_code == 400              # duplicate ids
    assert c.put("/api/v1/datasets/..%2Fx", content=good, headers=h(RUNNER)).status_code in (400, 404)
    assert c.get("/api/v1/datasets", headers=h(VIEWER)).json()["datasets"][0]["cases"] == 2


def _trace(i, version, error=False, when=None):
    when = when or datetime.now(timezone.utc)
    ns = int(when.timestamp() * 1e9)
    attrs = [{"key": "gen_ai.operation.name", "value": {"stringValue": "invoke_agent"}},
             {"key": "input.value", "value": {"stringValue": f"question {i}"}}]
    if not error:
        attrs.append({"key": "output.value", "value": {"stringValue": f"answer {i}"}})
    return {"resourceSpans": [{"resource": {"attributes": [{"key": "service.version", "value": {"stringValue": version}}]},
            "scopeSpans": [{"spans": [{"traceId": f"{i:032x}", "spanId": f"{i + 1:016x}", "name": "agent",
                                       "startTimeUnixNano": str(ns), "endTimeUnixNano": str(ns + 2_000_000_000),
                                       "attributes": attrs, "status": {"code": 2 if error else 1}}]}]}]}


def test_otlp_receiver_json_and_protobuf_and_roles(server):
    c, app, tmp = server
    assert c.post("/v1/traces", json=_trace(1, "1.0"), headers=h(VIEWER)).status_code == 403
    r = c.post("/v1/traces", json=_trace(1, "1.0"), headers=h(INGEST))
    assert r.status_code == 200 and r.json()["spans"] == 1
    assert c.post("/v1/traces", content=b"not json", headers={**h(INGEST), "content-type": "application/json"}).status_code == 400
    sdk = pytest.importorskip("opentelemetry.exporter.otlp.proto.common.trace_encoder")
    from tests.test_otel import _real_spans
    pb = sdk.encode_spans(_real_spans()).SerializeToString()
    r = c.post("/v1/traces", content=pb, headers={**h(INGEST), "content-type": "application/x-protobuf"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-protobuf")
    stored = app.state.storage.list("otel/")
    assert len(stored) == 2


def test_monitoring_detects_regression_between_windows(server, tmp_path):
    from evalkit.monitor import ingest, monitor
    c, app, root = server
    cfg = root / "mon.yaml"
    cfg.write_text(yaml.safe_dump({"name": "prod", "storage": {"type": "local", "root": str(root / "runs")},
                                   "system": {"type": "python", "target": "tests.test_server:echo_agent"},
                                   "monitoring": {"window_hours": 1, "min_n": 20, "min_delta": 0.1,
                                                  "metrics": ["error", "latency_s"]}}), encoding="utf-8")
    t0 = datetime.now(timezone.utc)
    for i in range(40):
        ingest(app.state.storage, _trace(i, "1.0", when=t0 - timedelta(minutes=10)))
    w1 = monitor(str(cfg), now=t0, log=lambda *a: None)
    assert w1["n"] == 40 and w1["overall"]["error"]["mean"] == 0.0 and not w1["alerts"]
    t1 = t0 + timedelta(hours=1)
    for i in range(100, 140):
        ingest(app.state.storage, _trace(i, "1.1", error=i % 2 == 0, when=t1 - timedelta(minutes=5)))
    w2 = monitor(str(cfg), now=t1, log=lambda *a: None)
    alert = {a["metric"]: a for a in w2["alerts"]}
    assert "error" in alert and alert["error"]["current"] == 0.5 and alert["error"]["p_value"] < 0.001
    assert w2["failure_patterns"]["errors"]
    hist = c.get("/api/v1/monitoring", headers=h(VIEWER)).json()["windows"]
    assert len(hist) == 2 and "1.1" in hist[-1]["by_version"]
    assert "## Alerts" in (Path(w2["run_dir"]) / "report.md").read_text(encoding="utf-8")


# ---------- OIDC ----------
@pytest.fixture(scope="module")
def idp():
    from cryptography.hazmat.primitives.asymmetric import rsa
    import jwt
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="k1", use="sig", alg="RS256")

    def token(**claims):
        base = {"iss": "https://idp.example", "aud": "api://evalkit", "exp": int(time.time()) + 600,
                "preferred_username": "jane@corp.example", "roles": ["EvalRunners"]}
        return jwt.encode({**base, **claims}, key, algorithm="RS256", headers={"kid": "k1"})
    return token, {"keys": [jwk]}


def _oidc_app(tmp_path, jwks):
    return create_app({"storage": {"type": "local", "root": str(tmp_path / "r")}, "configs_dir": str(tmp_path),
                       "session_secret": "x" * 40,
                       "auth": {"mode": "oidc", "oidc": {
                           "issuer": "https://idp.example", "audience": "api://evalkit", "client_id": "evalkit-ui",
                           "roles_claim": "roles", "role_map": {"EvalRunners": "runner", "EvalAdmins": "admin"},
                           "jwks": jwks, "discovery": {"authorization_endpoint": "https://idp.example/authorize",
                                                       "token_endpoint": "https://idp.example/token",
                                                       "jwks_uri": "https://idp.example/keys"}}}})


def test_oidc_bearer_tokens(idp, tmp_path):
    token, jwks = idp
    c = TestClient(_oidc_app(tmp_path, jwks))
    bearer = lambda t: {"authorization": f"Bearer {t}"}
    me = c.get("/api/v1/me", headers=bearer(token())).json()
    assert me["name"] == "jane@corp.example" and me["roles"] == ["runner"]
    assert c.get("/api/v1/runs", headers=bearer(token())).status_code == 200
    assert c.get("/api/v1/runs", headers=bearer(token(aud="someone-else"))).status_code == 401
    assert c.get("/api/v1/runs", headers=bearer(token(exp=int(time.time()) - 3600))).status_code == 401
    assert c.get("/api/v1/runs", headers=bearer(token(iss="https://evil.example"))).status_code == 401
    assert c.get("/api/v1/audit", headers=bearer(token())).status_code == 403            # runner, not admin
    assert c.get("/api/v1/audit", headers=bearer(token(roles=["EvalAdmins"]))).status_code == 200
    assert c.get("/api/v1/runs", headers=bearer(token(roles=["Unmapped"]))).status_code == 403
    r = c.get("/auth/login", follow_redirects=False)
    assert r.status_code in (302, 307) and "code_challenge_method=S256" in r.headers["location"]


def test_oidc_login_flow_checks_state_and_nonce(idp):
    token, jwks = idp
    v = OidcVerifier(issuer="https://idp.example", client_id="evalkit-ui", audience="api://evalkit", jwks=jwks,
                     roles_claim="roles", role_map={"EvalRunners": "runner"},
                     discovery={"authorization_endpoint": "https://idp.example/authorize",
                                "token_endpoint": "https://idp.example/token", "jwks_uri": "x"})
    session = {}
    url = v.login_url("https://evalkit.example/auth/callback", session)
    assert "state=" in url and session["pkce"]
    id_token = token(aud="evalkit-ui", nonce=session["nonce"])

    class Http:
        def post(self, url, data, timeout):
            assert data["code_verifier"] and data["grant_type"] == "authorization_code"
            return type("R", (), {"status_code": 200, "json": lambda self: {"id_token": id_token}})()
    with pytest.raises(AccessDeniedError, match="state"):
        v.complete_login("code", "forged-state", "https://evalkit.example/auth/callback", dict(session), http=Http())
    p = v.complete_login("code", session["state"], "https://evalkit.example/auth/callback", session, http=Http())
    assert p.name == "jane@corp.example" and p.roles == {"runner"}


def test_oidc_requires_session_secret_and_none_mode_is_local_only(tmp_path):
    with pytest.raises(ConfigError, match="session_secret"):
        create_app({"auth": {"mode": "oidc", "oidc": {"issuer": "https://idp.example", "jwks": {"keys": []}}},
                    "storage": {"type": "local", "root": str(tmp_path)}})
    from evalkit.server.app import serve
    with pytest.raises(ConfigError, match="local development"):
        serve(None, host="0.0.0.0")
