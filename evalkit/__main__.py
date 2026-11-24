"""evalkit command line.

  python -m evalkit run CONFIG [--limit N]       run, score, report; exit 1 if a gate fails
  python -m evalkit smoke CONFIG [--n 3]         run a few cases and print raw results
  python -m evalkit generate CONFIG [--n 30] [--unanswerable 5] [--out FILE]
  python -m evalkit rescore RUN_DIR              re-score saved results (no system calls)
  python -m evalkit compare RUN_A RUN_B          side-by-side metric changes
  python -m evalkit export-attacks --token T --out DIR   attack docs for a staging DB
  python -m evalkit list                         adapters, metrics, attacks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="evalkit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--limit", type=int)
    s = sub.add_parser("smoke"); s.add_argument("config"); s.add_argument("--n", type=int, default=3)
    g = sub.add_parser("generate"); g.add_argument("config"); g.add_argument("--n", type=int, default=30)
    g.add_argument("--unanswerable", type=int, default=5); g.add_argument("--out"); g.add_argument("--seed", type=int, default=0)
    rs = sub.add_parser("rescore"); rs.add_argument("run_dir")
    c = sub.add_parser("compare"); c.add_argument("run_a"); c.add_argument("run_b")
    e = sub.add_parser("export-attacks"); e.add_argument("--token", required=True)
    e.add_argument("--out", required=True); e.add_argument("--attacks", default="all")
    sub.add_parser("list")
    a = p.parse_args(argv)

    from . import runner

    if a.cmd == "run":
        report, run_dir = runner.run(a.config, limit=a.limit)
        runner.print_summary(report)
        print(f"\nReport: {run_dir / 'report.md'}")
        return 1 if report["gate_failures"] else 0

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

    if a.cmd == "compare":
        print(runner.compare(a.run_a, a.run_b)); return 0

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

    if a.cmd == "list":
        from . import metrics as M
        from .adapters import AGENTS, RETRIEVERS
        from .injection import ATTACK_PACK
        print("system types:   ", ", ".join(AGENTS), "+ custom")
        print("retriever types:", ", ".join(RETRIEVERS), "+ custom")
        print("metrics:        ", ", ".join(["task_success"] + list(M.REGISTRY)))
        print("attacks:        ", ", ".join(f"{x.id} ({x.category})" for x in ATTACK_PACK))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
