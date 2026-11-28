"""evalkit server: REST API, OTLP trace receiver and web dashboard.

    python -m evalkit serve --config server.yaml [--host 0.0.0.0 --port 8080]

server.yaml:
  storage: {type: local, root: /data/evalkit}     # or s3 / azure_blob / gcs / postgres
  configs_dir: /etc/evalkit/configs               # the ONLY eval configs the API may start
  datasets_dir: /data/datasets                    # uploaded datasets (versioned by content hash)
  auth: {...}                                     # see evalkit/server/auth.py
  session_secret: ${secret:evalkit-session-key}   # signs dashboard sessions
  public_url: https://evalkit.corp.example        # for the OIDC redirect URI
  max_parallel_runs: 2
  max_upload_mb: 20

Every endpoint except /healthz and /readyz requires authentication; every
state-changing call is written to the audit log with the caller's identity.
Configs can import Python classes, so the API never accepts a config body:
it only starts configs that an administrator placed in configs_dir.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from .. import __version__, audit
from ..errors import AccessDeniedError, ConfigError, EvalkitError
from .auth import Authenticator, Principal

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STATIC = Path(__file__).parent / "static"


def load_server_config(path: str | None) -> dict:
    from dotenv import load_dotenv

    from .. import secrets
    load_dotenv()
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    cfg = cfg or {}
    secrets.configure(cfg.get("secrets"))
    return cfg


class Jobs:
    """Background evaluation runs (bounded pool), visible via the API."""

    def __init__(self, workers: int):
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="evalkit-run")
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def submit(self, fn, actor: str, **info) -> str:
        jid = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[jid] = {"id": jid, "status": "queued", "submitted_by": actor,
                              "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}

        def wrapped():
            token = audit.set_actor(actor)
            self._set(jid, status="running")
            try:
                report, run_dir = fn()
                self._set(jid, status="finished", run_id=Path(run_dir).name,
                          gates="FAILED" if report and report.get("gate_failures") else "PASSED")
            except Exception as e:
                self._set(jid, status="failed", error=f"{type(e).__name__}: {e}"[:500])
            finally:
                audit._actor.reset(token)
        self.pool.submit(wrapped)
        return jid

    def _set(self, jid, **kw):
        with self.lock:
            self.jobs[jid].update(kw)

    def get(self, jid):
        with self.lock:
            return dict(self.jobs[jid]) if jid in self.jobs else None

    def list(self):
        with self.lock:
            return sorted((dict(j) for j in self.jobs.values()), key=lambda j: j["submitted_at"], reverse=True)


def create_app(cfg: dict | None = None):
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.sessions import SessionMiddleware
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
    from starlette.routing import Route

    from ..plugins import REGISTRY
    from ..storage import build_storage
    from ..storage.local import LocalStorage

    cfg = cfg or {}
    from ..secrets import resolve_config
    cfg = resolve_config(cfg)
    authn = Authenticator(cfg.get("auth"))
    storage = build_storage(cfg.get("storage"), default_root=cfg.get("output_dir", "results/evalkit"))
    local_root = (cfg.get("storage") or {}).get("root") if (cfg.get("storage") or {}).get("type", "local") == "local" \
        else cfg.get("output_dir", "results/evalkit")
    local = storage if isinstance(storage, LocalStorage) else LocalStorage(root=local_root or "results/evalkit")
    configs_dir = Path(cfg.get("configs_dir", "configs"))
    datasets_dir = Path(cfg.get("datasets_dir", "datasets"))
    max_upload = int(cfg.get("max_upload_mb", 20)) * 1024 * 1024
    jobs = Jobs(cfg.get("max_parallel_runs", 2))
    public_url = (cfg.get("public_url") or "").rstrip("/")
    session_secret = cfg.get("session_secret") or ""
    if authn.mode == "oidc" and len(session_secret) < 32:
        raise ConfigError("session_secret (>= 32 chars, from a secrets manager) is required with OIDC")
    session_secret = session_secret or uuid.uuid4().hex * 2

    # ---------- helpers ----------
    def who(request: Request) -> Principal:
        session = request.scope.get("session")
        return authn.authenticate({k.lower(): v for k, v in request.headers.items()}, session)

    def guarded(perm: str):
        def deco(fn):
            async def handler(request: Request):
                try:
                    p = who(request)
                    p.require(perm)
                except AccessDeniedError as e:
                    status = 401 if "required" in str(e) or "invalid" in str(e) or "unknown API key" in str(e) else 403
                    return JSONResponse({"error": str(e)}, status_code=status)
                token = audit.set_actor(f"{p.name} ({p.via})")
                try:
                    return await fn(request, p)
                except (ConfigError, ValueError) as e:
                    return JSONResponse({"error": str(e)}, status_code=400)
                except EvalkitError as e:
                    return JSONResponse({"error": str(e)}, status_code=422)
                finally:
                    audit._actor.reset(token)
            return handler
        return deco

    def safe_name(name: str, what: str) -> str:
        if not _NAME.match(name or "") or ".." in name:
            raise ValueError(f"invalid {what} name")
        return name

    def run_meta(rid: str) -> dict:
        text = local.read_text(f"{rid}/meta.json") or storage.read_text(f"{rid}/meta.json")
        meta = json.loads(text) if text else {"run_id": rid}
        rep = local.read_text(f"{rid}/report.json") or storage.read_text(f"{rid}/report.json")
        if rep:
            r = json.loads(rep)
            meta["gates"] = "FAILED" if r.get("gate_failures") else "PASSED"
            meta["gate_failures"] = r.get("gate_failures") or []
            meta["timestamp"] = r.get("timestamp")
            meta.setdefault("name", r.get("name"))
            meta["cases"] = r.get("cases")
            meta["headline"] = {k: (r.get("overall", {}).get(k) or {}).get("mean")
                                for k in ("task_success", "correctness", "faithfulness", "injection_resisted",
                                          "access_violation", "error", "latency_s", "total_tokens", "abstention")}
        meta.setdefault("name", rid.rsplit("-", 2)[0] if rid.count("-") >= 2 else rid)
        return meta

    # ---------- public ----------
    async def healthz(request):
        return JSONResponse({"status": "ok", "version": __version__})

    async def readyz(request):
        st = storage.health_check()
        return JSONResponse({"status": "ready" if st.ok else "not-ready", "storage": st.to_dict()},
                            status_code=200 if st.ok else 503)

    # ---------- runs ----------
    @guarded("read")
    async def list_runs(request, p):
        ids = sorted(set(local.list_runs()) | set(storage.list_runs()), reverse=True)
        ids = [i for i in ids if not i.startswith(("_", "monitoring", "otel"))]
        return JSONResponse({"runs": [run_meta(i) for i in ids[:500]]})

    @guarded("read")
    async def get_run(request, p):
        rid = safe_name(request.path_params["run_id"], "run")
        rep = local.read_text(f"{rid}/report.json") or storage.read_text(f"{rid}/report.json")
        if rep is None and not (local.read_text(f"{rid}/meta.json") or storage.read_text(f"{rid}/meta.json")):
            return JSONResponse({"error": "run not found"}, status_code=404)
        meta = run_meta(rid)
        res = local.root / rid / "results.jsonl"
        if res.is_file():
            keys = set()
            for line in res.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    keys.add((r["case_id"], r["repeat"]))
                except (ValueError, KeyError):
                    continue          # a line still being written
            done = len(keys)
        else:
            done = len(storage.list(f"{rid}/records/"))
        meta["progress"] = {"done": done, "planned": meta.get("planned")}
        return JSONResponse({"meta": meta, "report": json.loads(rep) if rep else None})

    @guarded("read")
    async def get_artifact(request, p):
        rid = safe_name(request.path_params["run_id"], "run")
        name = request.path_params["name"]
        if name not in ("report.md", "report.json", "access_report.json", "scores.jsonl", "meta.json"):
            return JSONResponse({"error": "unknown artifact"}, status_code=404)
        text = local.read_text(f"{rid}/{name}") or storage.read_text(f"{rid}/{name}")
        if text is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return PlainTextResponse(text, media_type="text/markdown" if name.endswith(".md") else "application/json")

    def _lines(text):
        return [json.loads(l) for l in (text or "").splitlines() if l.strip()]

    @guarded("read")
    async def run_cases(request, p):
        """Per-case results for the case explorer: input, answer, scores, trace, sources."""
        rid = safe_name(request.path_params["run_id"], "run")
        read = lambda k: local.read_text(f"{rid}/{k}") or storage.read_text(f"{rid}/{k}")
        cases = {c["id"]: c for c in _lines(read("cases.jsonl"))}
        if not cases:
            return JSONResponse({"error": "run not found"}, status_code=404)
        scores = {(r["case_id"], r["repeat"]): r["scores"] for r in _lines(read("scores.jsonl"))}
        rows = _lines(read("results.jsonl")) or [json.loads(t) for k in storage.list(f"{rid}/records/")
                                                 if (t := storage.read_text(k))]
        latest = {}
        for r in rows:
            latest[(r["case_id"], r["repeat"])] = r
        from ..privacy import redact_pii
        from ..secrets import redact
        red = bool((cfg.get("privacy") or {}).get("redact_ui", (cfg.get("privacy") or {}).get("redact_reports")))
        clean = (lambda t: redact_pii(redact(t or ""))[0]) if red else (lambda t: redact(t or ""))
        out = []
        for (cid, rep_), r in latest.items():
            c, res = cases.get(cid, {"id": cid}), r.get("result") or {}
            out.append({
                "case_id": cid, "repeat": rep_, "input": clean(c.get("input", "")), "tags": c.get("tags", []),
                "identity": c.get("identity"), "attack": (c.get("attack") or {}).get("id"),
                "expected": clean(str(c.get("expected_output") or c.get("expected_contains") or
                                      c.get("expected_number") or ("(unanswerable)" if c.get("unanswerable") else ""))),
                "answer": clean(res.get("answer", ""))[:6000], "error": res.get("error"), "status": res.get("status"),
                "latency_s": res.get("latency_s"),
                "tokens": (res.get("prompt_tokens") or 0) + (res.get("completion_tokens") or 0),
                "trace": [{"tool": t.get("tool"), "args": clean(json.dumps(t.get("args"), ensure_ascii=False))[:800],
                           "result": clean(t.get("result") or "")[:800]} for t in res.get("trace") or []][:30],
                "retrieved": [{"id": d.get("id"), "score": d.get("score"),
                               "source": (d.get("metadata") or {}).get("source")} for d in res.get("retrieved") or []][:20],
                "scores": scores.get((cid, rep_), {}),
            })
        return JSONResponse({"run_id": rid, "cases": out})

    @guarded("read")
    async def compare_api(request, p):
        a = safe_name(request.query_params.get("a", ""), "run")
        b = safe_name(request.query_params.get("b", ""), "run")
        for rid in (a, b):
            if not (local.root / rid / "report.json").is_file():
                return JSONResponse({"error": f"run '{rid}' has no local report"}, status_code=404)
        from ..runner import compare_runs
        alpha = float(request.query_params.get("alpha", 0.05))
        return JSONResponse({"a": run_meta(a), "b": run_meta(b), "alpha": alpha,
                             "metrics": compare_runs(str(local.root / a), str(local.root / b), alpha)})

    @guarded("run")
    async def start_run(request, p):
        body = await request.json()
        name = safe_name(body.get("config", ""), "config")
        path = (configs_dir / f"{name}.yaml")
        if not path.is_file():
            return JSONResponse({"error": f"no config '{name}' in configs_dir"}, status_code=404)
        limit = body.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer")
        from .. import runner
        overrides = {"output_dir": str(local.root)}
        if cfg.get("storage") and cfg["storage"].get("type", "local") != "local":
            overrides["storage"] = cfg["storage"]
        rid = runner.new_run_id({"name": name})
        jid = jobs.submit(lambda: runner.run(str(path), limit=limit, run_id=rid, overrides=overrides,
                                             log=lambda *a: None), actor=f"{p.name} ({p.via})",
                          config=name, run_id=rid)
        audit.emit("api.run.submitted", config=name, run_id=rid, job=jid)
        return JSONResponse({"job": jid, "run_id": rid}, status_code=202)

    @guarded("run")
    async def resume_run(request, p):
        rid = safe_name(request.path_params["run_id"], "run")
        run_dir = local.root / rid
        if not (run_dir / "config.yaml").is_file():
            return JSONResponse({"error": "run not found locally"}, status_code=404)
        from .. import runner
        jid = jobs.submit(lambda: runner.run(None, resume=str(run_dir), log=lambda *a: None),
                          actor=f"{p.name} ({p.via})", run_id=rid)
        audit.emit("api.run.resume", run_id=rid, job=jid)
        return JSONResponse({"job": jid, "run_id": rid}, status_code=202)

    @guarded("delete")
    async def delete_run(request, p):
        rid = safe_name(request.path_params["run_id"], "run")
        n = local.delete_run(rid) + (storage.delete_run(rid) if storage is not local else 0)
        audit.emit("api.run.deleted", run_id=rid, objects=n)
        return JSONResponse({"deleted": n})

    @guarded("read")
    async def list_jobs(request, p):
        return JSONResponse({"jobs": jobs.list()})

    @guarded("read")
    async def get_job(request, p):
        j = jobs.get(request.path_params["job_id"])
        return JSONResponse(j) if j else JSONResponse({"error": "job not found"}, status_code=404)

    @guarded("read")
    async def list_configs(request, p):
        names = sorted(f.stem for f in configs_dir.glob("*.yaml")) if configs_dir.is_dir() else []
        return JSONResponse({"configs": names})

    @guarded("read")
    async def plugins(request, p):
        return JSONResponse({"plugins": [{"category": s.category, "name": s.name, "status": s.status,
                                          "extra": s.extra, "description": s.description}
                                         for s in REGISTRY.list()]})

    # ---------- datasets (versioned by content hash) ----------
    @guarded("read")
    async def list_datasets(request, p):
        out = []
        if datasets_dir.is_dir():
            for f in sorted(datasets_dir.glob("*.jsonl")):
                name, _, ver = f.stem.partition("@")
                out.append({"name": name, "version": ver or None, "file": f.name,
                            "cases": sum(1 for l in f.read_text(encoding="utf-8").splitlines() if l.strip())})
        return JSONResponse({"datasets": out})

    @guarded("datasets")
    async def upload_dataset(request, p):
        name = safe_name(request.path_params["name"], "dataset")
        body = await request.body()
        if len(body) > max_upload:
            return JSONResponse({"error": "dataset too large"}, status_code=413)
        from ..dataset import load_cases
        tmp = datasets_dir / f".upload-{uuid.uuid4().hex}.jsonl"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(body)
        try:
            cases = load_cases(tmp)            # validates JSON, required fields, duplicate ids
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"invalid dataset: {e}")
        version = hashlib.sha256(body).hexdigest()[:12]
        final = datasets_dir / f"{name}@{version}.jsonl"
        tmp.replace(final)
        audit.emit("api.dataset.uploaded", dataset=name, version=version, cases=len(cases))
        return JSONResponse({"name": name, "version": version, "cases": len(cases), "file": final.name}, status_code=201)

    # ---------- OpenTelemetry receiver (OTLP/HTTP) ----------
    @guarded("ingest")
    async def otlp_traces(request, p):
        body = await request.body()
        if len(body) > max_upload:
            return JSONResponse({"error": "payload too large"}, status_code=413)
        from ..monitor import ingest
        ctype = request.headers.get("content-type", "")
        try:
            res = ingest(storage, body if "protobuf" in ctype else body.decode("utf-8"),
                         prefix=cfg.get("traces_prefix", "otel/"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        audit.emit("otel.ingested", spans=res["spans"], traces=res["traces"])
        if "protobuf" in ctype:      # OTLP/HTTP spec: reply with the same encoding
            return Response(b"", media_type="application/x-protobuf")
        return JSONResponse({"partialSuccess": {}, **res})

    @guarded("read")
    async def monitoring_history(request, p):
        text = storage.read_text("monitoring/history.jsonl") or ""
        return JSONResponse({"windows": [json.loads(l) for l in text.splitlines() if l.strip()][-200:]})

    # ---------- audit ----------
    @guarded("audit")
    async def audit_log(request, p):
        path = audit.default_path()
        lines = path.read_text(encoding="utf-8").splitlines()[-500:] if path.exists() else []
        ok, msg = audit.verify(path)
        return JSONResponse({"chain_ok": ok, "detail": msg, "events": [json.loads(l) for l in lines]})

    # ---------- dashboard + browser SSO ----------
    async def index(request):
        return FileResponse(STATIC / "index.html")

    async def me(request):
        try:
            p = who(request)
        except AccessDeniedError:
            return JSONResponse({"authenticated": False, "mode": authn.mode}, status_code=401)
        return JSONResponse({"authenticated": True, "name": p.name, "roles": sorted(p.roles), "via": p.via,
                             "mode": authn.mode})

    def redirect_uri(request):
        return f"{public_url}/auth/callback" if public_url else str(request.url_for("auth_callback"))

    async def login(request):
        if authn.mode != "oidc":
            return RedirectResponse("/")
        return RedirectResponse(authn.oidc.login_url(redirect_uri(request), request.session))

    async def auth_callback(request):
        if authn.mode != "oidc":
            return RedirectResponse("/")
        q = request.query_params
        if q.get("error"):
            return PlainTextResponse(f"login failed: {q.get('error')}", status_code=401)
        try:
            p = authn.oidc.complete_login(q.get("code", ""), q.get("state", ""), redirect_uri(request), request.session)
        except AccessDeniedError as e:
            return PlainTextResponse(str(e), status_code=401)
        request.session.update({"user": p.name, "roles": sorted(p.roles), "exp": time.time() + 8 * 3600})
        audit.emit("auth.login", user=p.name, roles=sorted(p.roles))
        return RedirectResponse("/")

    async def logout(request):
        request.session.clear()
        return RedirectResponse("/")

    routes = [
        Route("/healthz", healthz), Route("/readyz", readyz),
        Route("/", index), Route("/api/v1/me", me),
        Route("/auth/login", login), Route("/auth/callback", auth_callback, name="auth_callback"),
        Route("/auth/logout", logout),
        Route("/api/v1/runs", list_runs, methods=["GET"]), Route("/api/v1/runs", start_run, methods=["POST"]),
        Route("/api/v1/runs/{run_id}", get_run, methods=["GET"]),
        Route("/api/v1/runs/{run_id}", delete_run, methods=["DELETE"]),
        Route("/api/v1/runs/{run_id}/resume", resume_run, methods=["POST"]),
        Route("/api/v1/runs/{run_id}/artifacts/{name}", get_artifact),
        Route("/api/v1/runs/{run_id}/cases", run_cases),
        Route("/api/v1/compare", compare_api),
        Route("/api/v1/jobs", list_jobs), Route("/api/v1/jobs/{job_id}", get_job),
        Route("/api/v1/configs", list_configs), Route("/api/v1/plugins", plugins),
        Route("/api/v1/datasets", list_datasets, methods=["GET"]),
        Route("/api/v1/datasets/{name}", upload_dataset, methods=["PUT", "POST"]),
        Route("/v1/traces", otlp_traces, methods=["POST"]),
        Route("/api/v1/monitoring", monitoring_history),
        Route("/api/v1/audit", audit_log),
    ]

    async def security_headers(scope, receive, send, app):
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                            (b"referrer-policy", b"no-referrer"),
                            (b"content-security-policy", b"default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                                         b"script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")]
                message["headers"] = headers
            await send(message)
        await app(scope, receive, send_wrapper)

    class SecurityHeaders:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.app(scope, receive, send)
            await security_headers(scope, receive, send, self.app)

    secure_cookie = public_url.startswith("https://")
    app = Starlette(routes=routes, middleware=[
        Middleware(SecurityHeaders),
        Middleware(SessionMiddleware, secret_key=session_secret, https_only=secure_cookie,
                   same_site="lax", max_age=8 * 3600, session_cookie="evalkit_session"),
    ])
    app.state.jobs, app.state.storage, app.state.local = jobs, storage, local
    return app


def serve(config_path: str | None, host: str = "127.0.0.1", port: int = 8080):
    import uvicorn
    cfg = load_server_config(config_path)
    mode = ((cfg.get("auth") or {}).get("mode") or "none")
    if mode == "none" and host not in ("127.0.0.1", "localhost", "::1"):
        raise ConfigError("auth.mode none is for local development only; refusing to listen on " + host)
    audit.emit("server.started", host=host, port=port, auth=mode)
    uvicorn.run(create_app(cfg), host=host, port=port, proxy_headers=True, forwarded_allow_ips="*",
                log_level="info")
