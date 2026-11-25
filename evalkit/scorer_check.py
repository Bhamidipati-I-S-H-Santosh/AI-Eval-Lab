"""Check the rule-based abstention scorer against hand-labeled answers.

    python -m evalkit.scorer_check            # precision / recall per split
    python -m evalkit.scorer_check --errors   # also list every disagreement

Labels (tests/fixtures/abstention_labels.jsonl):
  refuse  - the answer gives up: must count as abstention
  answer  - a real answer, even if it contains "not" / "doesn't": must not
  partial - answers the question but notes a missing detail: must not
            (on an answerable question that is not "giving up")
  ambiguous - labelers disagree on what the right label is: left out of the score
Splits: dev (tuning), heldout (hand-written), blind-v1 and blind-v2 (other
models; each scored blind once, then used to fix the scorer, so now tuning
data). After tuning on a blind split, the next honest number needs a new one."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .metrics import abstained

LABELS = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "abstention_labels.jsonl"


def load(path: Path = LABELS) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score(rows: list[dict], fn=abstained) -> dict:
    rows = [r for r in rows if r["label"] != "ambiguous"]
    tp = sum(fn(r["answer"]) and r["label"] == "refuse" for r in rows)
    fp = sum(fn(r["answer"]) and r["label"] != "refuse" for r in rows)
    fn_ = sum(not fn(r["answer"]) and r["label"] == "refuse" for r in rows)
    return {"n": len(rows), "agree": len(rows) - fp - fn_,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn_) if tp + fn_ else 1.0,
            "wrong": [r for r in rows if fn(r["answer"]) != (r["label"] == "refuse")]}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    rows = load()
    for split in dict.fromkeys(r.get("split") for r in rows):
        s = score([r for r in rows if r.get("split") == split])
        print(f"{split:8} agree {s['agree']}/{s['n']}  precision {s['precision']:.2f}  recall {s['recall']:.2f}")
        if "--errors" in argv:
            for r in s["wrong"]:
                print(f"   [{r['label']:7}] scorer={abstained(r['answer'])!s:5} {r['answer'][:95]}")


if __name__ == "__main__":
    main()
