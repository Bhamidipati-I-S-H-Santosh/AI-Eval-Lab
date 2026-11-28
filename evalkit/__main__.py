"""evalkit command line.

Dashboard
  python -m evalkit ui [--port 8765] [--stop]       open the local dashboard (runs open it automatically)

Running evaluations
  python -m evalkit run CONFIG [--limit N] [--no-ui]   run, score, report; exit 1 if a gate fails
  python -m evalkit run CONFIG --resume RUN         continue an interrupted run (dir or id)
  python -m evalkit resume RUN_DIR                  same, using the run's saved config
  python -m evalkit prepare CONFIG --run-id ID      freeze the cases for distributed workers
  python -m evalkit run CONFIG --run-id ID --shard 0/4 --no-score     one worker's slice
  python -m evalkit finalize RUN [--config CONFIG]  merge all shards, score, report, gate
  python -m evalkit smoke CONFIG [--n 3]            run a few cases and print raw results
  python -m evalkit generate CONFIG [--n 30] [--unanswerable 5] [--out FILE]
  python -m evalkit rescore RUN_DIR                 re-score saved results (no system calls)
  python -m evalkit retry-errors RUN_DIR            re-run only errored/missing runs, then re-score
  python -m evalkit compare RUN_A RUN_B [--fail-on-regression]   paired significance test; CI regression gate

Plugins and configuration
  python -m evalkit plugins list [--category C]     every integration and its verification status
  python -m evalkit plugins health CONFIG           probe the system, retriever, judge, storage
  python -m evalkit validate CONFIG                 check a config without running anything
  python -m evalkit list                            adapters, metrics, attacks

Server, traces and monitoring
  python -m evalkit serve [--config server.yaml] [--host H --port P]   REST API + OTLP receiver + dashboard
  python -m evalkit monitor CONFIG [--window-hours 24]   score sampled live traces, alert on regressions
  python -m evalkit otel import FILES... --out cases.jsonl          turn recorded traces into test cases

Security and operations
  python -m evalkit export-attacks --token T --out DIR   attack docs for a staging DB
  python -m evalkit purge CONFIG --older-than-days 30 [--dry-run]   retention
  python -m evalkit audit-verify [PATH]             check the audit log's hash chain
  python -m evalkit keygen                          new key for encrypted judge caches
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _shard(text: str | None):
    if not text:
        return None
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("--shard must look like 0/4")
    if not (0 <= k < n):
        raise argparse.ArgumentTypeError("--shard i/N needs 0 <= i < N")
    return k, n


def _ui_hook(a):
    """Open the local dashboard on the run as soon as it starts (not in CI / with --no-ui)."""
    if getattr(a, "no_ui", False) or getattr(a, "shard", None):
        return None
    from . import ui
    return None if ui.disabled() else (lambda run_id, root: ui.open_run(run_id, root))


def _finish(report, run_dir, runner) -> int:
    runner.print_summary(report)
    print(f"\nReport: {run_dir / 'report.md'}")
    return 1 if report["gate_failures"] else 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252 and crash on ❌ / –
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="evalkit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--limit", type=int)
    r.add_argument("--resume", help="run directory or id to continue")
    r.add_argument("--run-id", help="use this run id (shared by distributed workers)")
    r.add_argument("--shard", type=_shard, help="run only slice i of N, e.g. 0/4")
    r.add_argument("--no-score", action="store_true", help="record results only (shard workers)")
    r.add_argument("--no-ui", action="store_true", help="don't open the dashboard on this run")
    u = sub.add_parser("ui"); u.add_argument("--port", type=int); u.add_argument("--root", default="results/evalkit")
    u.add_argument("--configs", default="examples"); u.add_argument("--no-browser", action="store_true")
    u.add_argument("--stop", action="store_true")
    rr = sub.add_parser("resume"); rr.add_argument("run_dir"); rr.add_argument("--no-ui", action="store_true")
    pr = sub.add_parser("prepare"); pr.add_argument("config"); pr.add_argument("--run-id", required=True)
    fz = sub.add_parser("finalize"); fz.add_argument("run"); fz.add_argument("--config")
    s = sub.add_parser("smoke"); s.add_argument("config"); s.add_argument("--n", type=int, default=3)
    g = sub.add_parser("generate"); g.add_argument("config"); g.add_argument("--n", type=int, default=30)
    g.add_argument("--unanswerable", type=int, default=5); g.add_argument("--out"); g.add_argument("--seed", type=int, default=0)
    rs = sub.add_parser("rescore"); rs.add_argument("run_dir")
    re_ = sub.add_parser("retry-errors"); re_.add_argument("run_dir")
    c = sub.add_parser("compare"); c.add_argument("run_a"); c.add_argument("run_b")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--fail-on-regression", action="store_true",
                   help="exit 1 only if a metric got significantly worse (CI regression gate)")
    c.add_argument("--metrics", help="comma-separated metrics the regression gate watches (default: all)")
    pl = sub.add_parser("plugins"); pl_sub = pl.add_subparsers(dest="plugins_cmd", required=True)
    pll = pl_sub.add_parser("list"); pll.add_argument("--category")
    plh = pl_sub.add_parser("health"); plh.add_argument("config")
    v = sub.add_parser("validate"); v.add_argument("config")
    e = sub.add_parser("export-attacks"); e.add_argument("--token", required=True)
    e.add_argument("--out", required=True); e.add_argument("--attacks", default="all")
    pu = sub.add_parser("purge"); pu.add_argument("config"); pu.add_argument("--older-than-days", type=float, required=True)
    pu.add_argument("--dry-run", action="store_true")
    av = sub.add_parser("audit-verify"); av.add_argument("path", nargs="?")
    sub.add_parser("keygen")
    sv = sub.add_parser("serve"); sv.add_argument("--config"); sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    mo = sub.add_parser("monitor"); mo.add_argument("config"); mo.add_argument("--window-hours", type=float)
    ot = sub.add_parser("otel"); ot_sub = ot.add_subparsers(dest="otel_cmd", required=True)
    oti = ot_sub.add_parser("import"); oti.add_argument("files", nargs="+"); oti.add_argument("--out", required=True)
    sub.add_parser("list")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    from . import runner
    from .errors import EvalkitError

    try:
        return _dispatch(a, runner)
    except EvalkitError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


def _dispatch(a, runner) -> int:
    if a.cmd == "ui":
        from . import ui
        if a.stop:
            print(ui.stop(a.root)); return 0
        ui.serve_local(a.root, port=a.port or ui.DEFAULT_PORT, configs_dir=a.configs, browser=not a.no_browser)
        return 0

    if a.cmd == "run":
        report, run_dir = runner.run(a.config, limit=a.limit, resume=a.resume, run_id=a.run_id,
                                     shard=a.shard, score=not a.no_score, on_start=_ui_hook(a))
        if report is None:
            print(f"\nRecorded results for this worker in {run_dir}. Score all shards with: "
                  f"python -m evalkit finalize {run_dir.name} --config {a.config}")
            return 0
        return _finish(report, run_dir, runner)

    if a.cmd == "resume":
        return _finish(*runner.run(None, resume=a.run_dir, on_start=_ui_hook(a)), runner)

    if a.cmd == "prepare":
        _, run_dir = runner.run(a.config, run_id=a.run_id, score=False, execute=False)
        print(f"Workers: python -m evalkit run {a.config} --run-id {a.run_id} --shard i/N --no-score")
        return 0

    if a.cmd == "finalize":
        return _finish(*runner.finalize(a.run, config_path=a.config), runner)

    if a.cmd == "smoke":
        cfg = runner.load_config(a.config)
        system, retriever, judge = runner.build_from_config(cfg)
        cases = [c for c in runner.build_cases(cfg, retriever, judge) if not c.attack][: a.n]
        for c in cases:
            res = runner._safe_run(system, c)
            print(f"\n--- {c.id}: {c.input}")
            print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False)[:3000])
        system.close()
        print("\nCheck: is 'answer' filled? Are trace/retrieved/tokens filled where your system has them?")
        return 0

    if a.cmd == "generate":
        from .dataset import generate_cases, save_cases
        cfg = runner.load_config(a.config)
        _, retriever, judge = runner.build_from_config(cfg)
        if retriever is None or judge is None:
            print("generate needs 'retriever' and 'judge' in the config"); return 2
        cases = generate_cases(retriever, judge, n=a.n, unanswerable=a.unanswerable, seed=a.seed)
        out = a.out or cfg.get("dataset", {}).get("path") or "data/generated_cases.jsonl"
        save_cases(cases, out)
        print(f"Wrote {len(cases)} cases to {out}. Review them, delete bad ones, "
              f"and remove the 'needs_review' tag from ones you've checked.")
        return 0

    if a.cmd == "rescore":
        report, run_dir = runner.rescore(a.run_dir)
        runner.print_summary(report)
        return 1 if report["gate_failures"] else 0

    if a.cmd == "retry-errors":
        return _finish(*runner.retry_errors(a.run_dir), runner)

    if a.cmd == "compare":
        print(runner.compare(a.run_a, a.run_b, alpha=a.alpha))
        if a.fail_on_regression:
            regs = runner.regressions(a.run_a, a.run_b, a.metrics.split(",") if a.metrics else None, a.alpha)
            for r in regs:
                print(f"REGRESSION: {r['metric']} {r['a']} -> {r['b']} (p={r['p']})")
            print("Regression gate: " + ("FAILED" if regs else "PASSED (no significant regressions)"))
            return 1 if regs else 0
        return 0

    if a.cmd == "plugins":
        from .plugins import REGISTRY
        if a.plugins_cmd == "list":
            rows = REGISTRY.list(a.category)
            print(f"{'category':<10} {'name':<26} {'status':<12} {'extra':<16} description")
            for s in rows:
                print(f"{s.category:<10} {s.name:<26} {s.status:<12} {(s.extra or '-'):<16} {s.description}")
            print("\nverified = exercised against the real library/service; mock-tested = implemented "
                  "against the vendor's documented API and tested with a fake client only.")
            return 0
        from .health import check_config
        results = check_config(a.config)
        bad = 0
        for name, st in results.items():
            mark = "ok  " if st.ok else "FAIL"
            bad += not st.ok
            lat = f" ({st.latency_ms} ms)" if st.latency_ms is not None else ""
            print(f"[{mark}] {name:<12} {st.detail}{lat}")
        return 1 if bad else 0

    if a.cmd == "validate":
        cfg = runner.load_config(a.config)
        from .health import build_components
        build_components(cfg, close=True)
        print(f"{a.config}: valid (every plugin built and accepted its options)")
        egress = cfg.get("_egress") or []
        print("\nData egress (where evaluation data will be sent):")
        for comp, host in egress:
            print(f"  {comp:<10} -> {host}")
        if not egress:
            print("  none: everything runs in this process / on local files")
        allowed = (cfg.get("privacy") or {}).get("allowed_hosts")
        print("  allowlist: " + (", ".join(allowed) if allowed is not None else "not set (privacy.allowed_hosts)"))
        return 0

    if a.cmd == "export-attacks":
        from .injection import get_attacks, planted_payloads
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        ids = "all" if a.attacks == "all" else a.attacks.split(",")
        for atk, text in planted_payloads(get_attacks(ids), a.token):
            (out / f"evalkit_attack_{atk.id}.md").write_text(
                f"# Staff announcement\n\n{text}\n", encoding="utf-8")
        print(f"Wrote attack documents to {out}. Load them into a STAGING copy of the database only, "
              f"then run with  attacks: {{mode: external, token: {a.token}}}")
        return 0

    if a.cmd == "purge":
        runner.purge(a.config, a.older_than_days, dry_run=a.dry_run)
        return 0

    if a.cmd == "audit-verify":
        from . import audit
        ok, msg = audit.verify(a.path)
        print(("OK: " if ok else "TAMPERED: ") + msg)
        return 0 if ok else 1

    if a.cmd == "serve":
        from .server.app import serve
        serve(a.config, host=a.host, port=a.port)
        return 0

    if a.cmd == "monitor":
        from .monitor import monitor
        res = monitor(a.config, window_hours=a.window_hours)
        print(f"\nScored {res['n']} trace(s); {len(res['alerts'])} alert(s). Report: {res['run_dir']}/report.md")
        return 1 if res["alerts"] else 0

    if a.cmd == "otel":
        from .otel import load_traces, normalize_spans
        traces = normalize_spans(load_traces(a.files))
        rows = [t.to_case() for t in traces if t.question]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        warn = sum(1 for t in traces if t.warnings)
        print(f"{len(traces)} trace(s) -> {len(rows)} case(s) in {a.out}" + (f" ({warn} with warnings)" if warn else ""))
        print("Add expected answers to turn them into a golden set, or evaluate as recorded with "
              "system: {type: otel_replay, traces: ...}")
        return 0

    if a.cmd == "keygen":
        from cryptography.fernet import Fernet
        print(Fernet.generate_key().decode())
        print("Store it in your secrets manager and reference it as judge.cache_key: ${secret:<name>}",
              file=sys.stderr)
        return 0

    if a.cmd == "list":
        from . import metrics as M
        from .adapters import AGENTS, RETRIEVERS
        from .injection import ATTACK_PACK
        print("system types:   ", ", ".join(AGENTS), "+ custom")
        print("retriever types:", ", ".join(RETRIEVERS), "+ custom")
        print("metrics:        ", ", ".join(["task_success"] + list(M.REGISTRY)))
        print("attacks:        ", ", ".join(f"{x.id} ({x.category})" for x in ATTACK_PACK))
        print("all plugins:     python -m evalkit plugins list")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
