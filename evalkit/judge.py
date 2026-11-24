"""LLM client used for judging answers and generating test cases.
Any OpenAI-compatible endpoint. Results are cached on disk so re-scoring a
run costs nothing.

config:
  judge:
    base_url: ${LLM_BASE_URL}
    api_key: ${LLM_API_KEY}
    model: ${LLM_MODEL}
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path

from .utils import expand_env

JUDGE_SYSTEM = "You are a strict, careful grader. Reply with JSON only."


class Judge:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 cache_dir: str = ".evalkit_cache", temperature: float = 0.0, retries: int = 3, **_):
        from openai import OpenAI
        self.model = expand_env(model)
        self.client = OpenAI(base_url=expand_env(base_url) or None, api_key=expand_env(api_key) or "none")
        self.temperature, self.retries = temperature, retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def complete(self, prompt: str, system: str = JUDGE_SYSTEM, use_cache: bool = True) -> str:
        key = hashlib.sha256(f"{self.model}\n{system}\n{prompt}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.txt"
        if use_cache and path.exists():
            return path.read_text(encoding="utf-8")
        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
                text = (resp.choices[0].message.content or "").strip()
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt * 3)
        if use_cache:
            with self._lock:
                path.write_text(text, encoding="utf-8")
        return text

    def grade(self, prompt: str) -> dict:
        """Returns {"score": float 0..1, "reason": str}."""
        return parse_json_score(self.complete(prompt))


def parse_json_score(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        score = float(data.get("score", 0))
        return {"score": max(0.0, min(1.0, score)), "reason": str(data.get("reason", ""))}
    except (ValueError, TypeError):
        return {"score": 0.0, "reason": f"unparseable judge output: {text[:100]}"}


CORRECTNESS = """Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Does the answer state the same key facts as the reference? Extra correct detail
is fine. Missing, wrong, or contradicting facts are not.
Reply as JSON: {{"score": 1 or 0, "reason": "<one sentence>"}}"""

FAITHFULNESS = """Context the system had access to:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the context? An answer that
says it doesn't know counts as supported.
Reply as JSON: {{"score": 1 or 0, "reason": "<one sentence>"}}"""

RELEVANCE = """Question: {question}
Answer: {answer}

Does the answer directly address the question (whether or not it is correct)?
Reply as JSON: {{"score": 1 or 0, "reason": "<one sentence>"}}"""
