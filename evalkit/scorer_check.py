"""Check the abstention scorer against labeled answers.

    python -m evalkit.scorer_check                     # regex v4, per split
    python -m evalkit.scorer_check --errors            # also list disagreements
    python -m evalkit.scorer_check --methods regex-v4,regex-v5,judge,hybrid --by-style
    python -m evalkit.scorer_check --pending out.csv   # label disputes for a human

Labels (tests/fixtures/abstention_labels.jsonl):
  refuse    - the answer gives up: must count as abstention
  answer    - a real answer, even if it contains "not" / "doesn't": must not
  partial   - answers the question but notes a missing detail: must not
  ambiguous - labelers can't agree what is right: left out of the score
Two labelers: when `label` and `label_2` disagree on refuse vs not-refuse, the
row is "pending" (left out) until a human sets `label_human`.

Splits: dev (tuning), heldout (hand-written), blind-v1..v3 (other models; each
scored blind once, some then used to fix the regex), blind-v4 (two labelers).
Intervals are Wilson 95%: with ~12 refusals, recall is only known to +-0.25.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from .abstention import abstained, classify

LABELS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "abstention_labels.jsonl"
METHODS = ("regex-v4", "regex-v5", "judge", "hybrid")


def load(path: Path = LABELS) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def final_label(r: dict) -> str:
    """label_human wins; two labelers that disagree on refuse-vs-not -> 'pending'."""
    if r.get("label_human"):
        return r["label_human"]
    if r.get("label_2") and (r["label_2"] == "refuse") != (r["label"] == "refuse"):
        return "pending"
    return r["label"]


def style(r: dict) -> str:
    src = r.get("source", "")
    if "prompt)" in src:
        return src.rsplit(",", 1)[-1].replace("prompt)", "").strip()
    return "agent eval run" if "eval run" in src else "hand-written"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(rows: list[dict], fn=None) -> dict:
    """fn(row) -> True if the scorer says the answer gave up (default: regex v4)."""
    fn = fn or (lambda r: abstained(r["answer"]))
    rows = [r for r in rows if final_label(r) not in ("ambiguous", "pending")]
    pred = [bool(fn(r)) for r in rows]
    gold = [final_label(r) == "refuse" for r in rows]
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn_ = sum(g and not p for p, g in zip(pred, gold))
    return {"n": len(rows), "agree": len(rows) - fp - fn_, "refusals": tp + fn_, "caught": tp,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn_) if tp + fn_ else 1.0,
            "precision_ci": wilson(tp, tp + fp), "recall_ci": wilson(tp, tp + fn_),
            "wrong": [r for r, p, g in zip(rows, pred, gold) if p != g]}


def make_judge(model: str):
    from .judge import Judge
    return Judge(model=model, base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))


def predictor(method: str, judge=None):
    def fn(r):
        q = r.get("question")
        return classify(q, r["answer"], judge=judge, method=method)["label"] == "refuse"
    if method == "regex-v5":
        return lambda r: classify(r.get("question"), r["answer"], judge=None, method="regex")["label"] == "refuse"
    return fn


def _fmt(s: dict) -> str:
    (pl, ph), (rl, rh) = s["precision_ci"], s["recall_ci"]
    return (f"agree {s['agree']:>3}/{s['n']:<3}  caught {s['caught']:>2}/{s['refusals']:<2}  "
            f"P {s['precision']:.2f} [{pl:.2f}-{ph:.2f}]  R {s['recall']:.2f} [{rl:.2f}-{rh:.2f}]")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--methods", default="regex-v4")
    ap.add_argument("--judge-model", default=os.getenv("ABSTENTION_JUDGE_MODEL", "openai/gpt-oss-20b"))
    ap.add_argument("--by-style", action="store_true")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--pending", metavar="CSV", help="write label disputes to CSV for a human")
    a = ap.parse_args(argv)
    rows = load()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    judge = None
    if any(m in ("judge", "hybrid") for m in methods):
        from dotenv import load_dotenv
        load_dotenv(LABELS.parents[2] / ".env")
        judge = make_judge(a.judge_model)

    groups = [("split", r.get("split")) for r in rows]
    keys = list(dict.fromkeys(k for _, k in groups))
    if a.by_style:
        keys += [f"style:{s}" for s in dict.fromkeys(style(r) for r in rows)]
    for m in methods:
        fn = predictor(m, judge)
        print(f"\n== {m}" + (f" (judge: {a.judge_model})" if m in ("judge", "hybrid") else ""))
        for k in keys + ["ALL"]:
            sel = (rows if k == "ALL" else [r for r in rows if style(r) == k[6:]] if k.startswith("style:")
                   else [r for r in rows if r.get("split") == k])
            s = score(sel, fn)
            print(f"  {k:24} {_fmt(s)}")
            if a.errors and k != "ALL" and not k.startswith("style:"):
                for r in s["wrong"]:
                    print(f"      [{final_label(r):7}] {r['answer'][:90]!r}")
    pend = [r for r in rows if final_label(r) == "pending"]
    if pend:
        print(f"\n{len(pend)} answers pending a human label (labelers disagree)")
    if a.pending:
        with open(a.pending, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["split", "question", "answer", "label", "label_2", "label_human (refuse/answer/partial)"])
            for r in pend:
                w.writerow([r.get("split"), r.get("question", ""), r["answer"], r["label"], r["label_2"], ""])
        print("wrote", a.pending)


if __name__ == "__main__":
    main()
