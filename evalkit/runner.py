"""Load a config, run every test case against the system, score, report, gate."""
from __future__ import annotations

import json
import shutil
import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from . import metrics as M
from .adapters import build_agent, build_retriever
from .adapters.base import RetrieverAsAgent
from .dataset import generate_cases, load_cases, save_cases
from .injection import PoisonedRetriever, build_attack_cases, build_planted_cases, get_attacks
from .judge import Judge
from .types import AgentResult, TestCase
from .utils import expand_env, import_object


def load_config(path: str | Path) -> dict:
    from dotenv import load_dotenv
    load_dotenv()
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_path"] = str(path)
    return cfg


# ---------- setup ----------
def build_from_config(cfg: dict):
    for m in cfg.get("custom_metrics", []):  # importing registers them
        import_object(m) if ":" in m else __import__(m)
    retriever = build_retriever(cfg["retriever"]) if cfg.get("retriever") else None
    if cfg.get("system"):
        system = build_agent(cfg["system"])
    elif retriever is not None:  # no system: evaluate the database by itself
        system = RetrieverAsAgent(retriever, k=cfg.get("run", {}).get("k", 5))
    else:
        raise ValueError("config needs 'system' or 'retriever'")
    judge = Judge(**expand_env(cfg["judge"])) if cfg.get("judge") else None
    return system, retriever, judge


def build_cases(cfg: dict, retriever, judge) -> list[TestCase]:
    ds = cfg.get("dataset", {})
    cases: list[TestCase] = []
    if ds.get("path") and Path(ds["path"]).exists():
        cases = load_cases(ds["path"])
    elif ds.get("generate"):
        g = ds["generate"]
        if retriever is None or judge is None:
            raise ValueError("dataset.generate needs both 'retriever' and 'judge' configured")
        cases = generate_cases(retriever, judge, n=g.get("n", 50),
                               unanswerable=g.get("unanswerable", 0), seed=g.get("seed", 0))
        out = ds.get("path") or g.get("save_to")
        if out:
            save_cases(cases, out)
            print(f"Generated {len(cases)} cases -> {out}  (review them before trusting scores)")
    else:
        raise ValueError("dataset needs an existing 'path' or a 'generate' block")
    if ds.get("tags"):
        cases = [c for c in cases if set(ds["tags"]) & set(c.tags)]
    if ds.get("limit"):
        cases = cases[: ds["limit"]]
    return cases


# ---------- running ----------
def _safe_run(system, case: TestCase) -> AgentResult:
    t0 = time.time()
    try:
        r = system.run(case.input)
    except Exception as e:
        r = AgentResult(answer="", error=f"{type(e).__name__}: {e}",
                        raw=traceback.format_exc(limit=3))
    if r.latency_s is None:
        r.latency_s = round(time.time() - t0, 3)
    return r


def run_cases(system, cases, retriever, attack_cfg: dict, repeats: int, concurrency: int,
              log=print) -> list[dict]:
    normal = [c for c in cases if not c.attack]
    attacks = [c for c in cases if c.attack]
    jobs = [(c, i) for c in normal for i in range(repeats)]
    rows: list[dict] = []

    def do(job):
        c, i = job
        return {"case_id": c.id, "repeat": i, "result": _safe_run(system, c).to_dict()}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for n, row in enumerate(pool.map(do, jobs), 1):
            rows.append(row)
            if n % 10 == 0 or n == len(jobs):
                log(f"  ran {n}/{len(jobs)}")

    mode = attack_cfg.get("mode", "wrapper")
    if attacks and mode == "wrapper":
        if not system.supports_retriever_injection or retriever is None:
            log("  ! skipping attack cases: wrapper mode needs a 'retriever' in the config and a "
                "system that supports set_retriever (use mode: mcp or external otherwise)")
            attacks = []
    # Attack cases run one at a time because swapping the retriever is global state.
    for c in attacks:
        for i in range(repeats):
            if mode == "wrapper":
                system.set_retriever(PoisonedRetriever(retriever, c.attack["payload"],
                                                       position=attack_cfg.get("position", 0)))
            try:
                rows.append({"case_id": c.id, "repeat": i, "result": _safe_run(system, c).to_dict()})
            finally:
                if mode == "wrapper":
                    system.set_retriever(None)  # restore the system's own search
    if attacks:
        log(f"  ran {len(attacks) * repeats} attack runs ({mode} mode)")
    return rows


# ---------- scoring ----------
def score_rows(rows, cases, cfg, judge) -> list[dict]:
    by_id = {c.id: c for c in cases}
    cost = cfg.get("cost", {})
    ctx = M.Context(judge=judge, price_in_per_1m=cost.get("input_per_1m", 0.0),
                    price_out_per_1m=cost.get("output_per_1m", 0.0),
                    extra={"abstention": cfg.get("abstention") or {}})   # {method: judge|hybrid|regex, judge_model}
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
        return out

    tags = sorted({t for c in cases for t in c.tags})
    by_tag = {t: summarize([r for r in scored if t in by_id[r["case_id"]].tags]) for t in tags}

    # Flaky = same case passes on some repeats and fails on others.
    flaky = []
    for cid in {r["case_id"] for r in scored}:
        vals = {r["scores"].get("task_success") for r in scored if r["case_id"] == cid} - {None}
        if len(vals) > 1:
            flaky.append(cid)
    return {"overall": summarize(scored), "by_tag": by_tag, "flaky_cases": sorted(flaky)}


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
    lines += ["", "## Overall", "", "| Metric | Mean | n | Std across repeats |", "|---|---|---|---|"]
    for m, v in summary["overall"].items():
        arrow = "↑" if M.higher_is_better(m) else "↓"
        lines.append(f"| {m} {arrow} | {v['mean']} | {v['n']} | {v['std_across_repeats']} |")
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
    if fails:
        lines += ["", "## Failures to read first", ""]
        for r in fails:
            c, res = by_id[r["case_id"]], r["result"]
            bad = [k for k, v in r["scores"].items() if not k.startswith("_") and v is not None
                   and M.higher_is_better(k) and k not in ("latency_s",) and v < 0.5]
            lines += [f"### {c.id} (repeat {r['repeat']})", f"- **Input:** {c.input}",
                      f"- **Answer:** {(res['answer'] or res.get('error') or '')[:300]}",
                      f"- **Expected:** {c.expected_output or c.expected_contains or c.expected_number}",
                      f"- **Failed checks:** {', '.join(bad) or 'error'}",
                      f"- **Tools:** {[t['tool'] for t in res.get('trace', [])]}", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def print_summary(report: dict):
    print("\n=== Eval summary ===")
    for m, v in report["overall"].items():
        extra = f"  (±{v['std_across_repeats']})" if v["std_across_repeats"] else ""
        print(f"{m:>22}: {v['mean']}{extra}   n={v['n']}")
    if report["flaky_cases"]:
        print(f"{'flaky cases':>22}: {len(report['flaky_cases'])}")
    print("\nGates:", "PASSED" if not report["gate_failures"] else "FAILED")
    for f in report["gate_failures"]:
        print("  ❌", f)


# ---------- entry points ----------
def run(config_path: str, limit: int | None = None, log=print) -> tuple[dict, Path]:
    cfg = load_config(config_path)
    if limit:
        cfg.setdefault("dataset", {})["limit"] = limit
    system, retriever, judge = build_from_config(cfg)
    cases = build_cases(cfg, retriever, judge)

    attack_cfg = cfg.get("attacks") or {}
    if attack_cfg.get("enabled", bool(attack_cfg)):
        selected = get_attacks(attack_cfg.get("attacks", "all"))
        if attack_cfg.get("mode", "wrapper") == "wrapper":
            cases += build_attack_cases(cases, selected, per_attack=attack_cfg.get("cases_per_attack", 2),
                                        seed=attack_cfg.get("seed", 0), token=attack_cfg.get("token"))
        else:  # mcp / external: attacks were planted with a known token
            if not attack_cfg.get("token"):
                raise ValueError("attacks.token is required for mcp/external mode (use the same "
                                 "token you gave the MCP proxy or export-attacks)")
            cases += build_planted_cases(cases, selected, str(attack_cfg["token"]),
                                         n=attack_cfg.get("cases", 10), seed=attack_cfg.get("seed", 0))

    run_cfg = cfg.get("run", {})
    run_dir = Path(cfg.get("output_dir", "results/evalkit")) / \
        f"{cfg.get('name', 'eval')}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    log(f"Running {len(cases)} cases x {run_cfg.get('repeats', 1)} repeats -> {run_dir}")

    rows = run_cases(system, cases, retriever, attack_cfg, run_cfg.get("repeats", 1),
                     run_cfg.get("concurrency", 4), log)
    system.close()
    if retriever is not None:
        retriever.close()
    save_cases(cases, run_dir / "cases.jsonl")
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return score_and_report(cfg, run_dir, cases, rows, judge)


def score_and_report(cfg, run_dir: Path, cases, rows, judge):
    scored = score_rows(rows, cases, cfg, judge)
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps({"case_id": r["case_id"], "repeat": r["repeat"], "scores": r["scores"]}) + "\n")
    summary = aggregate(scored, cases)
    failures = check_gates(summary["overall"], cfg.get("gates"))
    report = write_report(run_dir, cfg, summary, scored, cases, failures)
    return report, run_dir


def rescore(run_dir: str):
    """Re-compute metrics on saved results without calling the system again
    (e.g. after adding a metric or changing the judge)."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    for m in cfg.get("custom_metrics", []):
        import_object(m) if ":" in m else __import__(m)
    judge = Judge(**expand_env(cfg["judge"])) if cfg.get("judge") else None
    cases = load_cases(run_dir / "cases.jsonl")
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    return score_and_report(cfg, run_dir, cases, rows, judge)


def compare(a: str, b: str) -> str:
    ra = json.loads((Path(a) / "report.json").read_text())
    rb = json.loads((Path(b) / "report.json").read_text())
    lines = [f"{'metric':>22} | {'A':>9} | {'B':>9} | change", "-" * 58]
    for m in sorted(set(ra["overall"]) | set(rb["overall"])):
        va, vb = ra["overall"].get(m, {}).get("mean"), rb["overall"].get(m, {}).get("mean")
        if va is None or vb is None:
            lines.append(f"{m:>22} | {str(va):>9} | {str(vb):>9} |")
            continue
        d = vb - va
        better = (d > 0) == M.higher_is_better(m) if m in M.REGISTRY or m == "task_success" else None
        mark = "" if abs(d) < 1e-9 else ("better" if better else "WORSE")
        lines.append(f"{m:>22} | {va:>9} | {vb:>9} | {d:+.4f} {mark}")
    return "\n".join(lines)
