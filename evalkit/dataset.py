"""Load test cases, or generate them from any retriever that can sample docs."""
from __future__ import annotations

import json
from pathlib import Path

from .adapters.base import RetrieverAdapter
from .judge import Judge, parse_json_score  # noqa: F401  (re-exported for convenience)
from .types import TestCase

GEN_PROMPT = """You are writing test questions for a search assistant.

Passage (id: {doc_id}):
\"\"\"{text}\"\"\"

Write ONE question that a real user might ask and that this passage answers
fully. Phrase it naturally; do NOT copy distinctive phrases from the passage
(test understanding, not keyword matching). Then give the short correct answer
from the passage.

If the passage has no concrete facts worth asking about (e.g. a heading, a
table of contents, boilerplate), reply {{"skip": true}}.

Reply as JSON: {{"question": "...", "answer": "...", "key_terms": ["exact", "facts", "from", "answer"]}}"""

UNANSWERABLE_PROMPT = """Here are sample passages from a document collection:
{samples}

Write {n} questions a user of this collection might plausibly ask that are NOT
answered anywhere in these passages and are unlikely to be answered elsewhere
in the collection. Reply as JSON: {{"questions": ["...", "..."]}}"""


def load_cases(path: str | Path) -> list[TestCase]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    cases = []
    for i, row in enumerate(rows):
        row.setdefault("id", f"case{i:04d}")
        if row.get("expected_output") == "UNANSWERABLE":
            row["expected_output"], row["unanswerable"] = None, True
        cases.append(TestCase.from_dict(row))
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {path}")
    return cases


def save_cases(cases: list[TestCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in cases),
                    encoding="utf-8")


def _json(text: str) -> dict:
    import re
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def generate_cases(retriever: RetrieverAdapter, llm: Judge, n: int = 50,
                   unanswerable: int = 0, seed: int = 0, min_chars: int = 120) -> list[TestCase]:
    """Sample passages and have an LLM write one question per passage.
    Every generated case is tagged 'generated' and 'needs_review': read and
    prune them before trusting the scores."""
    docs = [d for d in retriever.sample(n * 2, seed=seed) if len(d.text) >= min_chars][:n]
    cases: list[TestCase] = []
    for d in docs:
        data = _json(llm.complete(GEN_PROMPT.format(doc_id=d.id, text=d.text[:3000]),
                                  system="You write high-quality evaluation data. Reply with JSON only."))
        if data.get("skip") or not data.get("question") or not data.get("answer"):
            continue
        cases.append(TestCase(
            id=f"gen-{len(cases):04d}", input=data["question"], expected_output=data["answer"],
            expected_sources=[d.id], tags=["generated", "needs_review"],
            metadata={"source_text": d.text[:500], "key_terms": data.get("key_terms", [])}))
    if unanswerable:
        samples = "\n---\n".join(d.text[:400] for d in docs[:8])
        data = _json(llm.complete(UNANSWERABLE_PROMPT.format(samples=samples, n=unanswerable),
                                  system="Reply with JSON only."))
        for q in (data.get("questions") or [])[:unanswerable]:
            cases.append(TestCase(id=f"gen-u{len(cases):04d}", input=q, unanswerable=True,
                                  tags=["generated", "needs_review", "unanswerable"]))
    return cases
