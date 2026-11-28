"""Run the golden set and score the system.

Retrieval metrics (no API key needed):
  hit_rate  - share of answerable questions where the right document is in the top-k
  mrr       - mean reciprocal rank of the first correct chunk (1.0 = always first)

Answer metrics (needs an LLM, graded by an LLM judge):
  correctness  - answer matches the reference answer
  faithfulness - every claim in the answer is supported by the retrieved context
  abstention   - on unanswerable questions, the system says "I don't know"

Examples:
  python -m src.evaluate --retrieval-only
  python -m src.evaluate --method hybrid --rerank
  python -m src.evaluate --retrieval-only --min-hit-rate 0.8   # CI gate
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from .ingest import build_chunks
from .retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden_set.jsonl"
RESULTS = ROOT / "results"

JUDGE_SYSTEM = "You are a strict grader. Reply with JSON only, no other text."

CORRECTNESS_PROMPT = """Question: {question}
Reference answer: {reference}
Model answer: {answer}

Does the model answer convey the same facts as the reference answer?
Minor wording differences are fine; missing or wrong facts are not.
Reply as JSON: {{"score": 1 or 0, "reason": "<short reason>"}}"""

FAITHFULNESS_PROMPT = """Context:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the context above?
An answer of "I don't know" counts as supported.
Reply as JSON: {{"score": 1 or 0, "reason": "<short reason>"}}"""


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_judge(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        return {"score": int(data.get("score", 0)), "reason": str(data.get("reason", ""))}
    except (ValueError, TypeError):
        return {"score": 0, "reason": f"unparseable judge output: {text[:80]}"}


def is_abstention(answer: str) -> bool:
    # Same scorer as evalkit, validated against hand labels (python -m evalkit.scorer_check).
    from evalkit.metrics import abstained
    return abstained(answer)


def run(args) -> dict:
    chunks = build_chunks(args.chunking, args.chunk_size, args.overlap)
    retriever = Retriever(chunks, method=args.method, rerank=args.rerank)
    golden = load_golden()

    rows = []
    for item in golden:
        t0 = time.time()
        row = {"id": item["id"], "question": item["question"]}
        answerable = item["source"] is not None

        if args.retrieval_only:
            retrieved = retriever.search(item["question"], top_k=args.top_k)
        else:
            from .rag import answer, format_context
            from .llm import chat

            ans, retrieved = answer(item["question"], retriever, top_k=args.top_k)
            row["answer"] = ans
            if answerable:
                j = parse_judge(chat(CORRECTNESS_PROMPT.format(
                    question=item["question"], reference=item["answer"], answer=ans),
                    system=JUDGE_SYSTEM))
                row["correct"], row["correct_reason"] = j["score"], j["reason"]
            else:
                row["abstained"] = int(is_abstention(ans))
            j = parse_judge(chat(FAITHFULNESS_PROMPT.format(
                context=format_context(retrieved), answer=ans), system=JUDGE_SYSTEM))
            row["faithful"], row["faithful_reason"] = j["score"], j["reason"]

        row["retrieved"] = [c.id for c in retrieved]
        if answerable:
            ranks = [i for i, c in enumerate(retrieved) if c.source == item["source"]]
            row["hit"] = int(bool(ranks))
            row["rr"] = 1.0 / (ranks[0] + 1) if ranks else 0.0
        row["latency_s"] = round(time.time() - t0, 2)
        rows.append(row)
        if args.verbose:
            print(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    def avg(key):
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": args.method,
        "rerank": args.rerank,
        "chunking": args.chunking,
        "top_k": args.top_k,
        "n_chunks": len(chunks),
        "hit_rate": avg("hit"),
        "mrr": avg("rr"),
        "correctness": avg("correct"),
        "faithfulness": avg("faithful"),
        "abstention": avg("abstained"),
        "avg_latency_s": avg("latency_s"),
    }
    return {"summary": summary, "rows": rows}


def save(result: dict) -> Path:
    runs = RESULTS / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = runs / f"{stamp}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    history = RESULTS / "history.csv"
    s = result["summary"]
    new = not history.exists()
    with history.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(s))
        if new:
            w.writeheader()
        w.writerow(s)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["bm25", "vector", "hybrid"], default="bm25")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--chunking", choices=["section", "fixed"], default="section")
    p.add_argument("--chunk-size", type=int, default=60)
    p.add_argument("--overlap", type=int, default=15)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--retrieval-only", action="store_true", help="skip the LLM; score retrieval only")
    p.add_argument("--min-hit-rate", type=float, default=None, help="exit 1 if hit rate is below this")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    result = run(args)
    s = result["summary"]
    print("\n=== Eval summary ===")
    for k, v in s.items():
        if v is not None:
            print(f"{k:>15}: {v}")

    misses = [r for r in result["rows"] if r.get("hit") == 0]
    if misses:
        print("\nRetrieval misses (look at these first):")
        for r in misses:
            print(f"  {r['id']}: {r['question']}  ->  {r['retrieved']}")

    if not args.no_save:
        print(f"\nSaved: {save(result).relative_to(ROOT)}")

    if args.min_hit_rate is not None and (s["hit_rate"] or 0) < args.min_hit_rate:
        print(f"\nFAIL: hit_rate {s['hit_rate']} < {args.min_hit_rate}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
