"""Load a config, run every test case against the system, score, report, gate.

Runs are durable: every finished case is written immediately (local file,
fsync'd, and mirrored to the configured storage backend). An interrupted run
resumes with  python -m evalkit run CONFIG --resume RUN_DIR_OR_ID : finished
cases are kept, errored and missing ones run again.

Distributed: workers sharing a storage backend each run a slice
(--run-id X --shard i/N --no-score); then  python -m evalkit finalize X  scores.
"""
from __future__ import annotations

import json
import logging
import statistics
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import engine
from . import metrics as M
from .adapters import build_agent, build_retriever
from .adapters.base import RetrieverAsAgent
from .dataset import generate_cases, load_cases, save_cases
from .errors import ConfigError
from .injection import PoisonedRetriever, build_attack_cases, build_planted_cases, get_attacks
from .judge import build_judge
from .storage import build_storage, record_key
from .storage.local import LocalStorage
from .types import AgentResult, TestCase
from .utils import import_object

log_ = logging.getLogger("evalkit.runner")


def load_config(path: str | Path) -> dict:
    from dotenv import load_dotenv

    from . import secrets
    load_dotenv()
    text = Path(path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: the config must be a YAML mapping")
    cfg["_path"] = str(path)
    cfg["_raw"] = text
    secrets.configure(cfg.get("secrets"))
    from .config import validate_config
    validate_config(cfg)
    from . import privacy
    privacy.install_log_redaction()
    cfg["_egress"] = privacy.check_egress(cfg)     # raises if privacy.allowed_hosts forbids a destination
    return cfg


# ---------- setup ----------
def build_from_config(cfg: dict, on_usage=None):
    for m in cfg.get("custom_metrics", []):  # importing registers them
        import_object(m) if ":" in m else __import__(m)
    retriever = build_retriever(cfg["retriever"]) if cfg.get("retriever") else None
    if cfg.get("system"):
        system = build_agent(cfg["system"])
    elif retriever is not None:  # no system: evaluate the database by itself
        system = RetrieverAsAgent(retriever, k=cfg.get("run", {}).get("k", 5))
    else:
        raise ConfigError("config needs 'system' or 'retriever'")
    judge = build_judge(cfg.get("judge"), on_usage=on_usage)
    return system, retriever, judge


def build_cases(cfg: dict, retriever, judge) -> list[TestCase]:
    ds = cfg.get("dataset", {})
    cases: list[TestCase] = []
    if ds.get("path") and Path(ds["path"]).exists():
        cases = load_cases(ds["path"])
    elif ds.get("generate"):
        g = ds["generate"]
        if retriever is None or judge is None:
            raise ConfigError("dataset.generate needs both 'retriever' and 'judge' configured")
        cases = generate_cases(retriever, judge, n=g.get("n", 50),
                               unanswerable=g.get("unanswerable", 0), seed=g.get("seed", 0))
        out = ds.get("path") or g.get("save_to")
        if out:
            save_cases(cases, out)
            print(f"Generated {len(cases)} cases -> {out}  (review them before trusting scores)")
    elif ds.get("path"):
        raise ConfigError(f"dataset.path '{ds['path']}' does not exist")
    else:
        raise ConfigError("dataset needs an existing 'path' or a 'generate' block")
    if ds.get("tags"):
        cases = [c for c in cases if set(ds["tags"]) & set(c.tags)]
    if ds.get("limit"):
        cases = cases[: ds["limit"]]
    return cases


def add_attack_cases(cfg: dict, cases: list[TestCase]) -> list[TestCase]:
    attack_cfg = cfg.get("attacks") or {}
    if not attack_cfg.get("enabled", bool(attack_cfg)):
        return cases
    selected = get_attacks(attack_cfg.get("attacks", "all"))
    if attack_cfg.get("mode", "wrapper") == "wrapper":
        return cases + build_attack_cases(cases, selected, per_attack=attack_cfg.get("cases_per_attack", 2),
                                          seed=attack_cfg.get("seed", 0), token=attack_cfg.get("token"))
    if not attack_cfg.get("token"):  # mcp / external: attacks were planted with a known token
        raise ConfigError("attacks.token is required for mcp/external mode (use the same "
                          "token you gave the MCP proxy or export-attacks)")
    return cases + build_planted_cases(cases, selected, str(attack_cfg["token"]),
                                       n=attack_cfg.get("cases", 10), seed=attack_cfg.get("seed", 0))


# ---------- running ----------
def _call(system, case: TestCase) -> AgentResult:
    if case.turns and len(case.turns) > 1:
        return system.run_conversation(list(case.turns))
    return system.run(case.input)


def _safe_run(system, case: TestCase, retry: engine.RetryPolicy | None = None,
              limiter: engine.RateLimiter | None = None, budget: engine.Budget | None = None,
              router=None) -> AgentResult:
    retry = retry or engine.RetryPolicy()
    lock = None
    if router is not None and case.identity:
        system, lock = router.for_identity(case.identity)
    attempt = 0
    while True:
        attempt += 1
        if limiter is not None:
            limiter.acquire()
        t0 = time.time()
        try:
            if lock is not None:
                with lock:
                    system.set_identity(case.identity)
                    r = _call(system, case)
            else:
                r = _call(system, case)
        except Exception as e:
            r = AgentResult(answer="", error=f"{type(e).__name__}: {e}",
                            raw=traceback.format_exc(limit=3))
        if r.latency_s is None:
            r.latency_s = round(time.time() - t0, 3)
        used = (r.prompt_tokens or 0) + (r.completion_tokens or 0)
        if limiter is not None:
            limiter.record_tokens(used)
        if budget is not None:
            budget.charge(r.prompt_tokens or 0, r.completion_tokens or 0)
        if retry.should_retry(r.error, attempt):
            time.sleep(retry.delay(attempt))
            continue
        return r


class _Recorder:
    """Writes each finished row at once: local (fsync) + optional mirror."""

    def __init__(self, local: LocalStorage, run_id: str, mirror=None, log=print, redactor=None):
        self.local, self.run_id, self.mirror, self.log = local, run_id, mirror, log
        self.redactor = redactor if (redactor is not None and redactor.stored) else None
        self.mirror_failures = 0

    def __call__(self, row: dict) -> None:
        key = record_key(row["case_id"], row["repeat"])
        row = {**row, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if self.redactor is not None:
            row["result"] = self.redactor.result(row["result"])
        self.local.append_record(self.run_id, key, row)
        if self.mirror is not None:
            try:
                self.mirror.append_record(self.run_id, key, row)
            except Exception as e:  # kept locally; synced again at finalize
                self.mirror_failures += 1
                if self.mirror_failures <= 3:
                    self.log(f"  ! storage mirror write failed ({type(e).__name__}); kept locally")


def _in_shard(case_id: str, repeat: int, shard: tuple[int, int] | None) -> bool:
    if not shard:
        return True
    import zlib
    k, n = shard
    return zlib.crc32(f"{case_id}#{repeat}".encode()) % n == k


def run_cases(system, cases, retriever, attack_cfg: dict, repeats: int, concurrency: int,
              log=print, done: set | None = None, persist=None, limiter=None, budget=None,
              retry=None, shard: tuple[int, int] | None = None, stop: dict | None = None,
              router=None) -> list[dict]:
    """Run every (case, repeat) not in `done`. Each finished row is passed to
    `persist` immediately. If the budget runs out, remaining jobs are skipped
    (stop['reason'] is set) and the run can be resumed later."""
    done = done or set()
    stop = stop if stop is not None else {}
    lock = threading.Lock()

    def mine(c, i):
        return (c.id, i) not in done and _in_shard(c.id, i, shard)

    normal = [(c, i) for c in cases if not c.attack for i in range(repeats) if mine(c, i)]
    attacks = [(c, i) for c in cases if c.attack for i in range(repeats) if mine(c, i)]
    rows: list[dict] = []

    def finish(c, i, res: AgentResult):
        row = {"case_id": c.id, "repeat": i, "result": res.to_dict()}
        with lock:
            rows.append(row)
        if persist:
            persist(row)
        if budget is not None and not stop.get("reason"):
            reason = budget.exceeded()
            if reason:
                stop["reason"] = reason
                log(f"  ! {reason}: stopping; resume later with --resume")

    def do(job):
        c, i = job
        if stop.get("reason"):
            return None
        finish(c, i, _safe_run(system, c, retry, limiter, budget, router))
        return True

    if normal:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(do, j) for j in normal]
            for n, f in enumerate(as_completed(futures), 1):
                f.result()
                if n % 10 == 0 or n == len(normal):
                    log(f"  ran {n}/{len(normal)}")

    mode = attack_cfg.get("mode", "wrapper")
    if attacks and mode == "wrapper":
        if not system.supports_retriever_injection or retriever is None:
            log("  ! skipping attack cases: wrapper mode needs a 'retriever' in the config and a "
                "system that supports set_retriever (use mode: mcp or external otherwise)")
            attacks = []
    # Attack cases run one at a time because swapping the retriever is global state.
    for c, i in attacks:
        if stop.get("reason"):
            break
        if mode == "wrapper":
            system.set_retriever(PoisonedRetriever(retriever, c.attack["payload"],
                                                   position=attack_cfg.get("position", 0)))
        try:
            res = _safe_run(system, c, retry, limiter, budget, router)
        finally:
            if mode == "wrapper":
                system.set_retriever(None)  # restore the system's own search
        finish(c, i, res)
    if attacks:
        log(f"  ran {len(attacks)} attack runs ({mode} mode)")
    return rows


# ---------- scoring ----------
def _identity_ctx(cfg: dict) -> dict:
    icfg = cfg.get("identity") or {}
    if not icfg.get("acl"):
        return {}
    from .identity import build_acl, load_users
    return {"acl": build_acl(icfg), "users": load_users(icfg), "unknown": icfg.get("unknown", "report")}


def score_rows(rows, cases, cfg, judge) -> list[dict]:
    by_id = {c.id: c for c in cases}
    cost = cfg.get("cost", {})
    ctx = M.Context(judge=judge, price_in_per_1m=cost.get("input_per_1m", 0.0),
                    price_out_per_1m=cost.get("output_per_1m", 0.0),
                    extra={"abstention": cfg.get("abstention") or {},   # {method: judge|hybrid|regex, judge_model}
                           "identity": _identity_ctx(cfg)})
    names = cfg.get("metrics") or (["task_success"] + list(M.REGISTRY))
    unknown = [n for n in names if n != "task_success" and n not in M.REGISTRY]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; known: {sorted(M.REGISTRY)}")
    if judge is None:
        names = [n for n in names if n == "task_success" or not M.REGISTRY[n].needs_judge]

    def do(row):
        case = by_id[row["case_id"]]
        return {**row, "scores": M.score_case(case, AgentResult.from_dict(row["result"]), ctx, names)}

    with ThreadPoolExecutor(max_workers=cfg.get("run", {}).get("judge_concurrency", 4)) as pool:
        return list(pool.map(do, rows))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(scored: list[dict], cases: list[TestCase]) -> dict:
    by_id = {c.id: c for c in cases}
    metric_names = sorted({k for r in scored for k in r["scores"] if not k.startswith("_")})
    repeats = sorted({r["repeat"] for r in scored})

    def summarize(rows):
        out = {}
        for m in metric_names:
            vals = [r["scores"].get(m) for r in rows]
            mean = _mean(vals)
            if mean is None:
                continue
            per_rep = [_mean([r["scores"].get(m) for r in rows if r["repeat"] == i]) for i in repeats]
            per_rep = [p for p in per_rep if p is not None]
            out[m] = {"mean": mean, "n": sum(v is not None for v in vals),
                      "std_across_repeats": round(statistics.pstdev(per_rep), 4) if len(per_rep) > 1 else 0.0}
            ci = confidence_interval(rows, m)
            if ci:
                out[m]["ci95"] = ci
        return out

    tags = sorted({t for c in cases for t in c.tags})
    by_tag = {t: summarize([r for r in scored if t in by_id[r["case_id"]].tags]) for t in tags}

    # Errors (rate limits, timeouts) count as failures in task_success. This
    # second number shows quality on the runs that actually got an answer.
    ok = [r for r in scored if not (r.get("result") or {}).get("error")]
    ts_ok = _mean([r["scores"].get("task_success") for r in ok])
    excl = None
    if ts_ok is not None and len(ok) < len(scored):
        excl = {"mean": ts_ok, "n": sum(r["scores"].get("task_success") is not None for r in ok),
                "errored_runs": len(scored) - len(ok)}

    # Flaky = same case passes on some repeats and fails on others.
    flaky = []
    for cid in {r["case_id"] for r in scored}:
        vals = {r["scores"].get("task_success") for r in scored if r["case_id"] == cid} - {None}
        if len(vals) > 1:
            flaky.append(cid)
    out = {"overall": summarize(scored), "by_tag": by_tag, "flaky_cases": sorted(flaky)}
    if excl:
        out["task_success_excl_errors"] = excl
    return out


# Metrics that are not 0..1 rates get no interval.
_NOT_RATES = {"latency_s", "cost_usd", "total_tokens", "steps"}


def _case_means(rows, m) -> dict:
    """Average each case over its repeats, so repeats don't pose as extra evidence."""
    per: dict = {}
    for r in rows:
        v = r["scores"].get(m)
        if v is not None:
            per.setdefault(r["case_id"], []).append(v)
    return {cid: sum(v) / len(v) for cid, v in per.items()}


def confidence_interval(rows, m) -> list[float] | None:
    """95% interval over cases: Wilson for pass/fail metrics, normal
    approximation (clipped to 0..1) when repeats make case scores fractional."""
    if m in _NOT_RATES:
        return None
    vals = list(_case_means(rows, m).values())
    n = len(vals)
    if n == 0 or any(v < 0 or v > 1 for v in vals):
        return None
    from .utils import wilson
    if all(v in (0.0, 1.0) for v in vals):
        lo, hi = wilson(int(sum(vals)), n)
    else:
        mean = sum(vals) / n
        sd = statistics.stdev(vals) if n > 1 else 0.0
        half = 1.96 * sd / n ** 0.5
        lo, hi = max(0.0, mean - half), min(1.0, mean + half)
    return [round(lo, 3), round(hi, 3)]


def check_gates(overall: dict, gates: dict) -> list[str]:
    """gates: {metric: ">=0.85"} or {metric: 0.85} (means >=). Returns failures."""
    failures = []
    for name, rule in (gates or {}).items():
        rule = str(rule).strip()
        op, val = (rule[:2], rule[2:]) if rule[:2] in (">=", "<=") else (">=", rule)
        actual = overall.get(name, {}).get("mean")
        if actual is None:
            failures.append(f"{name}: no data (metric not computed)")
        elif (op == ">=" and actual < float(val)) or (op == "<=" and actual > float(val)):
            failures.append(f"{name}: {actual} (needs {op} {val})")
    return failures


# ---------- reports ----------
def write_report(run_dir: Path, cfg: dict, summary: dict, scored: list[dict], cases, failures):
    by_id = {c.id: c for c in cases}
    report = {"name": cfg.get("name", "eval"), "timestamp": datetime.now().isoformat(timespec="seconds"),
              "cases": len(cases), "gate_failures": failures, **summary}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"# Eval report: {report['name']}", "", f"- Run: {report['timestamp']}",
             f"- Cases: {len(cases)}, runs: {len(scored)}",
             f"- Gates: {'PASSED' if not failures else 'FAILED'}", ""]
    for f in failures:
        lines.append(f"  - ❌ {f}")
    lines += ["", "## Overall", "", "| Metric | Mean | 95% CI | n | Std across repeats |", "|---|---|---|---|---|"]
    for m, v in summary["overall"].items():
        arrow = "↑" if M.higher_is_better(m) else "↓"
        ci = f"{v['ci95'][0]}–{v['ci95'][1]}" if v.get("ci95") else "–"
        lines.append(f"| {m} {arrow} | {v['mean']} | {ci} | {v['n']} | {v['std_across_repeats']} |")
    ex = summary.get("task_success_excl_errors")
    if ex:
        lines += ["", f"Task success excluding {ex['errored_runs']} errored runs: **{ex['mean']}** "
                      f"(n={ex['n']}). Re-run them with `python -m evalkit retry-errors {run_dir}`."]
    lines += ["", "95% CIs are computed over cases (repeats averaged first). With few cases they "
                  "are wide: a difference inside them may be noise."]
    key = [m for m in ("task_success", "correctness", "source_recall", "injection_resisted")
           if m in summary["overall"]]
    if summary["by_tag"] and key:
        lines += ["", "## By tag", "", "| Tag | " + " | ".join(key) + " |", "|---" * (len(key) + 1) + "|"]
        for t, v in summary["by_tag"].items():
            lines.append(f"| {t} | " + " | ".join(str(v.get(m, {}).get("mean", "–")) for m in key) + " |")
    if summary["flaky_cases"]:
        lines += ["", "## Flaky cases (different results across repeats)", ""]
        lines += [f"- {c}" for c in summary["flaky_cases"]]
    fails = [r for r in scored if r["scores"].get("task_success") == 0.0][:15]
    red = _redactor(cfg)
    show = red.text if red.reports else (lambda x: x)
    if fails:
        lines += ["", "## Failures to read first", ""]
        for r in fails:
            c, res = by_id[r["case_id"]], r["result"]
            bad = [k for k, v in r["scores"].items() if not k.startswith("_") and v is not None
                   and M.higher_is_better(k) and k not in ("latency_s",) and v < 0.5]
            lines += [f"### {c.id} (repeat {r['repeat']})", f"- **Input:** {show(c.input)}",
                      f"- **Answer:** {show((res['answer'] or res.get('error') or '')[:300])}",
                      f"- **Expected:** {show(str(c.expected_output or c.expected_contains or c.expected_number))}",
                      f"- **Failed checks:** {', '.join(bad) or 'error'}",
                      f"- **Tools:** {[t['tool'] for t in res.get('trace', [])]}", ""]
    if any(r["scores"].get("_access") is not None for r in scored):
        from .identity import access_report
        acc = access_report(scored, cases)
        (run_dir / "access_report.json").write_text(json.dumps(acc, indent=2), encoding="utf-8")
        report["access"] = {"per_identity": acc["per_identity"], "violations": len(acc["violations"])}
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines += ["", "## Access control (identity-aware)", "",
                  "| Identity | Runs | Runs with a violation | Runs with unknown ACLs |", "|---|---|---|---|"]
        for ident, v in acc["per_identity"].items():
            lines.append(f"| {ident} | {v['runs']} | {v['violations']} | {v['unknown']} |")
        if acc["violations"]:
            lines += ["", "### Violations (a document the identity may not read was exposed)", "",
                      "| Question | Identity | Document | Evidence |", "|---|---|---|---|"]
            for v in acc["violations"][:50]:
                lines.append(f"| {v['case']} | {v['identity']} | {v['doc']} | {v['evidence']} |")
        else:
            lines += ["", "No access violations found."]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def print_summary(report: dict):
    print("\n=== Eval summary ===")
    for m, v in report["overall"].items():
        extra = f"  (±{v['std_across_repeats']})" if v["std_across_repeats"] else ""
        ci = f"  [{v['ci95'][0]}-{v['ci95'][1]}]" if v.get("ci95") else ""
        print(f"{m:>22}: {v['mean']}{ci}{extra}   n={v['n']}")
    ex = report.get("task_success_excl_errors")
    if ex:
        print(f"{'excl. errors':>22}: task_success {ex['mean']} ({ex['errored_runs']} errored runs; "
              f"retry with: python -m evalkit retry-errors RUN_DIR)")
    if report["flaky_cases"]:
        print(f"{'flaky cases':>22}: {len(report['flaky_cases'])}")
    print("\nGates:", "PASSED" if not report["gate_failures"] else "FAILED")
    for f in report["gate_failures"]:
        print("  ❌", f)


# ---------- run lifecycle ----------
def _sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _storages(cfg: dict, root: str | Path | None = None):
    """(local working storage, optional mirror). Reports always land locally."""
    st = cfg.get("storage") or {}
    output_dir = str(root or cfg.get("output_dir", "results/evalkit"))
    if st.get("type") == "local":
        return LocalStorage(root=str(root or st.get("root", output_dir))), None
    return LocalStorage(root=output_dir), (build_storage(st) if st else None)


def dedupe(rows: list[dict]) -> list[dict]:
    """Last record per (case, repeat) wins; first-seen order is kept."""
    out: dict[tuple, dict] = {}
    for r in rows:
        out[(r["case_id"], r["repeat"])] = r
    return list(out.values())


def _merged_records(local: LocalStorage, mirror, run_id: str) -> list[dict]:
    rows = list(local.read_records(run_id))
    if mirror is not None:
        seen = {(r["case_id"], r["repeat"], r.get("recorded_at")) for r in rows}
        remote = [r for r in mirror.read_records(run_id)
                  if (r["case_id"], r["repeat"], r.get("recorded_at")) not in seen]
        rows = sorted(rows + remote, key=lambda r: r.get("recorded_at") or "")
    return rows


def _write_artifact(local: LocalStorage, mirror, key: str, text: str, log=print) -> None:
    local.write_text(key, text)
    if mirror is not None:
        try:
            mirror.write_text(key, text)
        except Exception as e:
            log(f"  ! could not write {key} to storage: {type(e).__name__}: {e}")


def _read_meta(local: LocalStorage, mirror, run_id: str) -> dict:
    text = local.read_text(f"{run_id}/meta.json") or (mirror.read_text(f"{run_id}/meta.json") if mirror else None)
    return json.loads(text) if text else {}


def _redactor(cfg: dict):
    from .privacy import Redactor
    return Redactor(cfg.get("privacy"))


def new_run_id(cfg: dict) -> str:
    return f"{cfg.get('name', 'eval')}-{datetime.now():%Y%m%d-%H%M%S}"


def _resolve_ref(ref: str | Path | None) -> tuple[Path | None, str | None]:
    """A run reference is a run directory or a bare run id."""
    if ref is None:
        return None, None
    p = Path(ref)
    if p.is_dir():
        return p.parent, p.name
    return None, str(ref)


def run(config_path: str | None = None, limit: int | None = None, log=print, resume: str | None = None,
        run_id: str | None = None, shard: tuple[int, int] | None = None, score: bool = True,
        execute: bool = True, overrides: dict | None = None, on_start=None):
    """Start a run, or continue one (resume=run dir or id). Returns (report, run_dir);
    report is None when score=False (shard workers)."""
    from . import audit
    root, resume_id = _resolve_ref(resume)
    if config_path is None:
        if root is None:
            raise ConfigError("resuming by run id needs the config path (it says where the run is stored)")
        config_path = str(root / resume_id / "config.yaml")
    cfg = load_config(config_path)
    if overrides:                       # e.g. the server forcing its own output_dir / storage
        cfg.update(overrides)
    if limit:
        cfg.setdefault("dataset", {})["limit"] = limit
    local, mirror = _storages(cfg, root)
    limiter, budget, retry = engine.from_config(cfg)
    system, retriever, judge = build_from_config(cfg, on_usage=budget.charge)
    rid = resume_id or run_id or new_run_id(cfg)

    try:
        cases_text = local.read_text(f"{rid}/cases.jsonl") or \
            (mirror.read_text(f"{rid}/cases.jsonl") if mirror else None)
        if cases_text:
            if not local.read_text(f"{rid}/cases.jsonl"):      # pulled from shared storage
                local.write_text(f"{rid}/cases.jsonl", cases_text)
                local.write_text(f"{rid}/config.yaml", cfg["_raw"])
            cases = [TestCase.from_dict(json.loads(l)) for l in cases_text.splitlines() if l.strip()]
            meta = _read_meta(local, mirror, rid)
            log(f"Continuing run {rid} ({len(cases)} cases)")
        else:
            if resume_id:
                raise ConfigError(f"run '{rid}' not found (no cases.jsonl in local or configured storage)")
            from .identity import expand_identity_cases
            cases = expand_identity_cases(cfg, add_attack_cases(cfg, build_cases(cfg, retriever, judge)))
            cases_text = "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in cases)
            meta = {"run_id": rid, "name": cfg.get("name", "eval"),
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "config_sha256": _sha256(cfg["_raw"]), "dataset_sha256": _sha256(cases_text),
                    "dataset_path": (cfg.get("dataset") or {}).get("path"),
                    "evalkit_version": _version(), "system_version": (cfg.get("system") or {}).get("version")}
            _write_artifact(local, mirror, f"{rid}/config.yaml", cfg["_raw"], log)
            _write_artifact(local, mirror, f"{rid}/cases.jsonl", cases_text, log)
            audit.emit("run.started", run_id=rid, config=cfg.get("_path"), cases=len(cases))
        if not execute:          # 'prepare': freeze the cases so shard workers share them
            _write_artifact(local, mirror, f"{rid}/meta.json", json.dumps({**meta, "status": "prepared"}, indent=2), log)
            log(f"Prepared run {rid}: {len(cases)} cases")
            return None, local.run_dir(rid)

        existing = dedupe(_merged_records(local, mirror, rid))
        planned = len(cases) * cfg.get("run", {}).get("repeats", 1)
        meta["planned"] = planned
        _write_artifact(local, mirror, f"{rid}/meta.json",
                        json.dumps({**meta, "status": "running", "planned": planned,
                                    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                   indent=2), log)
        if on_start is not None:          # e.g. the CLI opening the dashboard on this run
            try:
                on_start(rid, local.root)
            except Exception as e:
                log(f"  (could not open the dashboard: {type(e).__name__}: {e})")
        done = {(r["case_id"], r["repeat"]) for r in existing if not (r.get("result") or {}).get("error")}
        run_cfg = cfg.get("run", {})
        repeats = run_cfg.get("repeats", 1)
        log(f"Running {len(cases)} cases x {repeats} repeats -> {local.run_dir(rid)}"
            + (f"  ({len(done)} already done)" if done else "")
            + (f"  [shard {shard[0]}/{shard[1]}]" if shard else ""))
        stop: dict = {}
        router = None
        if (cfg.get("identity") or {}).get("users"):
            from .identity import IdentityRouter
            router = IdentityRouter(cfg, system)
        try:
            run_cases(system, cases, retriever, cfg.get("attacks") or {}, repeats,
                      run_cfg.get("concurrency", 4), log, done=done,
                      persist=_Recorder(local, rid, mirror, log, redactor=_redactor(cfg)),
                      limiter=limiter, budget=budget,
                      retry=retry, shard=shard, stop=stop, router=router)
        finally:
            if router is not None:
                router.close()
    finally:
        system.close()
        if retriever is not None:
            retriever.close()

    meta.update({"status": "incomplete" if stop.get("reason") else ("shard-complete" if shard else "complete"),
                 "stop_reason": stop.get("reason"),
                 "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "usage": budget.to_dict()})
    _write_artifact(local, mirror, f"{rid}/meta.json", json.dumps(meta, indent=2), log)
    audit.emit("run.executed", run_id=rid, status=meta["status"], reason=stop.get("reason"))
    if not score:
        return None, local.run_dir(rid)
    return finalize(rid, cfg=cfg, local=local, mirror=mirror, judge=judge, log=log)


def finalize(run_ref: str, config_path: str | None = None, log=print, cfg=None, local=None, mirror=None,
             judge=None):
    """Merge every recorded result (all shards), score, report, gate, sync."""
    from . import audit
    if cfg is None:
        root, rid = _resolve_ref(run_ref)
        config_path = config_path or (str(root / rid / "config.yaml") if root else None)
        if not config_path:
            raise ConfigError("finalize by run id needs the config path")
        cfg = load_config(config_path)
        local, mirror = _storages(cfg, root)
        if not local.read_text(f"{rid}/cases.jsonl") and mirror is not None:
            for k in ("cases.jsonl", "config.yaml", "meta.json"):
                t = mirror.read_text(f"{rid}/{k}")
                if t:
                    local.write_text(f"{rid}/{k}", t)
        judge = build_judge(cfg.get("judge"))
    else:
        rid = run_ref
    run_dir = local.run_dir(rid)
    rows = dedupe(_merged_records(local, mirror, rid))
    if not rows:
        raise ConfigError(f"run '{rid}' has no recorded results")
    # compact the local log (one line per case/repeat) and fill gaps in the mirror
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                           encoding="utf-8")
    if mirror is not None:
        have = {k.rsplit("/", 1)[-1].removesuffix(".json") for k in mirror.list(f"{rid}/records/")}
        for r in rows:
            k = record_key(r["case_id"], r["repeat"])
            if k not in have:
                try:
                    mirror.append_record(rid, k, r)
                except Exception as e:
                    log(f"  ! could not sync {k} to storage: {type(e).__name__}")
    cases = load_cases(run_dir / "cases.jsonl")
    report, run_dir = score_and_report(cfg, run_dir, cases, rows, judge)
    if mirror is not None:
        for k in ("report.json", "report.md", "scores.jsonl"):
            p = run_dir / k
            if p.exists():
                try:
                    mirror.write_text(f"{rid}/{k}", p.read_text(encoding="utf-8"))
                except Exception as e:
                    log(f"  ! could not write {k} to storage: {type(e).__name__}: {e}")
    audit.emit("run.finalized", run_id=rid, gates="FAILED" if report["gate_failures"] else "PASSED")
    return report, run_dir


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"


def score_and_report(cfg, run_dir: Path, cases, rows, judge):
    rows = dedupe(rows)
    scored = score_rows(rows, cases, cfg, judge)
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps({"case_id": r["case_id"], "repeat": r["repeat"], "scores": r["scores"]}) + "\n")
    summary = aggregate(scored, cases)
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        summary["run_meta"] = {k: meta.get(k) for k in ("run_id", "status", "stop_reason", "usage",
                                                         "dataset_sha256", "config_sha256", "evalkit_version",
                                                         "system_version")}
    failures = check_gates(summary["overall"], cfg.get("gates"))
    if (summary.get("run_meta") or {}).get("status") == "incomplete":
        failures.append(f"run incomplete: {summary['run_meta'].get('stop_reason')} (resume it with --resume)")
    report = write_report(run_dir, cfg, summary, scored, cases, failures)
    return report, run_dir


def rescore(run_dir: str):
    """Re-compute metrics on saved results without calling the system again
    (e.g. after adding a metric or changing the judge)."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    for m in cfg.get("custom_metrics", []):
        import_object(m) if ":" in m else __import__(m)
    judge = build_judge(cfg.get("judge"))
    cases = load_cases(run_dir / "cases.jsonl")
    rows = LocalStorage(root=str(run_dir.parent)).read_records(run_dir.name)
    return score_and_report(cfg, run_dir, cases, rows, judge)


def retry_errors(run_dir: str, log=print):
    """Re-run the errored (and any missing) runs of a saved run, keep every
    other result, then re-score. Same as resuming it."""
    return run(None, resume=run_dir, log=log)


def purge(cfg_path: str, older_than_days: float, dry_run: bool = False, log=print) -> list[str]:
    """Retention: delete runs older than N days from local and configured storage."""
    from . import audit
    cfg = load_config(cfg_path)
    local, mirror = _storages(cfg)
    cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
    deleted = []
    for store in [s for s in (local, mirror) if s is not None]:
        for rid in store.list_runs():
            meta_t = store.read_text(f"{rid}/meta.json")
            created = None
            if meta_t:
                try:
                    created = datetime.fromisoformat(json.loads(meta_t)["created_at"]).timestamp()
                except (ValueError, KeyError):
                    created = None
            if created is None:
                try:      # legacy runs: <name>-YYYYmmdd-HHMMSS
                    created = datetime.strptime(rid.rsplit("-", 2)[-2] + rid.rsplit("-", 1)[-1],
                                                "%Y%m%d%H%M%S").timestamp()
                except (ValueError, IndexError):
                    continue
            if created < cutoff:
                deleted.append(f"{type(store).__name__}:{rid}")
                if not dry_run:
                    store.delete_run(rid)
    if not dry_run and deleted:
        audit.emit("runs.purged", count=len(deleted), older_than_days=older_than_days)
    log(f"{'Would delete' if dry_run else 'Deleted'} {len(deleted)} run(s) older than {older_than_days} days")
    return deleted


def _load_scores(run_dir: Path) -> list[dict]:
    p = run_dir / "scores.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def compare_runs(a: str, b: str, alpha: float = 0.05) -> list[dict]:
    """Per-metric comparison of run B against baseline A with a paired sign
    test over the cases both runs share. verdict: better | WORSE | n.s. | same | ?"""
    from .utils import sign_test_p
    ra = json.loads((Path(a) / "report.json").read_text(encoding="utf-8"))
    rb = json.loads((Path(b) / "report.json").read_text(encoding="utf-8"))
    sa, sb = _load_scores(Path(a)), _load_scores(Path(b))
    out = []
    for m in sorted(set(ra["overall"]) | set(rb["overall"])):
        va, vb = ra["overall"].get(m, {}).get("mean"), rb["overall"].get(m, {}).get("mean")
        row = {"metric": m, "a": va, "b": vb, "delta": None, "wins": None, "losses": None, "n": None,
               "p": None, "verdict": None}
        if va is None or vb is None:
            out.append(row)
            continue
        d = vb - va
        row["delta"] = round(d, 6)
        known = m in M.REGISTRY or m == "task_success"
        hib = M.higher_is_better(m) if known else True
        ca, cb = _case_means(sa, m), _case_means(sb, m)
        common = set(ca) & set(cb)
        if not common:
            row["verdict"] = "same" if abs(d) < 1e-9 else "?"
        else:
            diffs = [(cb[c] - ca[c]) * (1 if hib else -1) for c in common]
            wins, losses = sum(x > 1e-9 for x in diffs), sum(x < -1e-9 for x in diffs)
            pv = sign_test_p(wins, losses)
            row.update(wins=wins, losses=losses, n=len(common), p=round(pv, 6))
            if wins == losses == 0:
                row["verdict"] = "same"
            elif pv < alpha and known:
                row["verdict"] = "better" if wins > losses else "WORSE"
            else:
                row["verdict"] = "n.s."
        out.append(row)
    return out


def compare(a: str, b: str, alpha: float = 0.05) -> str:
    """Side-by-side metrics. A change is called better/WORSE only if an exact
    paired sign test over the cases both runs share gives p < alpha; otherwise
    it is marked 'n.s.' (could be noise)."""
    lines = [f"{'metric':>22} | {'A':>9} | {'B':>9} | {'change':>8} | paired test (cases B better/worse)",
             "-" * 92]
    for r in compare_runs(a, b, alpha):
        if r["delta"] is None:
            lines.append(f"{r['metric']:>22} | {str(r['a']):>9} | {str(r['b']):>9} |")
            continue
        test = f"{r['wins']}/{r['losses']} of {r['n']}, p={r['p']:.3f}" if r["n"] else "no per-case scores to pair"
        mark = {"same": "", "?": "?"}.get(r["verdict"], r["verdict"])
        lines.append(f"{r['metric']:>22} | {r['a']:>9} | {r['b']:>9} | {r['delta']:+8.4f} | {test} {mark}")
    lines.append(f"\nbetter/WORSE = paired sign test p < {alpha}; n.s. = the change could be noise.\n"
                 "The test uses only cases scored in both runs; the means use every scored case "
                 "in each run, so they can disagree when runs errored on different cases.")
    return "\n".join(lines)


def regressions(baseline: str, candidate: str, metrics: list[str] | None = None, alpha: float = 0.05) -> list[dict]:
    """Metrics that got significantly worse (CI gate: noise never fails the build)."""
    return [r for r in compare_runs(baseline, candidate, alpha)
            if r["verdict"] == "WORSE" and (not metrics or r["metric"] in metrics)]
