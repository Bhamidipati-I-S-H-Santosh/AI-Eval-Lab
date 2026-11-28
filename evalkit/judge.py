"""LLM client used for judging answers and generating test cases.

Works with any model provider (evalkit.providers): OpenAI-compatible endpoints,
Azure OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, Ollama. Results are
cached on disk so re-scoring a run costs nothing.

config:
  judge:
    model: ${LLM_MODEL}
    base_url: ${LLM_BASE_URL}       # OpenAI-compatible (the default provider)
    api_key: ${LLM_API_KEY}
  # or any provider:
  judge:
    provider: bedrock
    model: anthropic.claude-3-5-sonnet-20240620-v1:0
    region: eu-west-1
    cache: encrypted                # plain (default) | encrypted | off
    cache_key: ${secret:evalkit-cache-key}   # Fernet key for 'encrypted'

The cache holds prompts and answers, which can contain customer data: use
'encrypted' or 'off' for sensitive datasets.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable

from .errors import ConfigError
from .utils import expand_env

log = logging.getLogger("evalkit.judge")

JUDGE_SYSTEM = "You are a strict, careful grader. Reply with JSON only."

_PROVIDER_KEYS = {"model", "cache_dir", "temperature", "retries", "provider", "cache", "cache_key",
                  "max_tokens", "on_usage"}


class Judge:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 cache_dir: str = ".evalkit_cache", temperature: float = 0.0, retries: int = 3,
                 provider: str | dict | object | None = None, cache: str = "plain",
                 cache_key: str | None = None, max_tokens: int | None = None,
                 on_usage: Callable[[int, int], None] | None = None, **provider_options):
        from .providers import ModelProvider, build_provider
        self.model = expand_env(model)
        if isinstance(provider, ModelProvider):
            self.provider = provider
        else:
            pcfg = provider if isinstance(provider, dict) else {"type": provider or "openai_compatible"}
            pcfg = {**provider_options, **pcfg}
            if pcfg["type"] == "openai_compatible":
                pcfg.setdefault("base_url", expand_env(base_url) if base_url else None)
                pcfg.setdefault("api_key", expand_env(api_key) if api_key else None)
            self.provider = build_provider(pcfg)
        self.temperature, self.retries, self.max_tokens = temperature, retries, max_tokens
        self.on_usage = on_usage
        self.cache_mode = (cache or "plain").lower()
        if self.cache_mode not in ("plain", "encrypted", "off"):
            raise ConfigError("judge.cache must be plain, encrypted or off")
        self._fernet = None
        if self.cache_mode == "encrypted":
            if not cache_key:
                raise ConfigError("judge.cache: encrypted needs judge.cache_key (a Fernet key; "
                                  "create one with: python -m evalkit keygen)")
            from cryptography.fernet import Fernet
            self._fernet = Fernet(expand_env(cache_key).encode())
        self.cache_dir = Path(cache_dir)
        if self.cache_mode != "off":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.prompt_tokens = self.completion_tokens = self.calls = 0

    # Kept for code that read judge.client (older custom metrics).
    @property
    def client(self):
        return getattr(self.provider, "client", None)

    def with_model(self, model: str, retries: int | None = None) -> "Judge":
        """Same provider and cache, different model."""
        j = object.__new__(Judge)
        j.__dict__.update(self.__dict__)
        j.model, j._lock = model, threading.Lock()
        j.retries = max(self.retries, retries or 0)
        return j

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / (f"{key}.enc" if self._fernet else f"{key}.txt")

    def _read_cache(self, key: str) -> str | None:
        if self.cache_mode == "off":
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        data = path.read_bytes()
        if self._fernet:
            try:
                return self._fernet.decrypt(data).decode("utf-8")
            except Exception:
                log.warning("judge cache entry %s could not be decrypted (wrong key?); ignoring it", path.name)
                return None
        return data.decode("utf-8")

    def _write_cache(self, key: str, text: str) -> None:
        if self.cache_mode == "off":
            return
        path = self._cache_path(key)
        data = self._fernet.encrypt(text.encode("utf-8")) if self._fernet else text.encode("utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            tmp.write_bytes(data)
            tmp.replace(path)          # atomic: no half-written cache entries

    def complete(self, prompt: str, system: str = JUDGE_SYSTEM, use_cache: bool = True) -> str:
        key = hashlib.sha256(f"{self.model}\n{system}\n{prompt}".encode()).hexdigest()
        if use_cache:
            hit = self._read_cache(key)
            if hit is not None:
                return hit
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        for attempt in range(self.retries):
            try:
                out = self.provider.complete(messages, self.model, temperature=self.temperature,
                                             max_tokens=self.max_tokens)
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt * 3)
        text = out.text
        with self._lock:
            self.calls += 1
            self.prompt_tokens += out.prompt_tokens or 0
            self.completion_tokens += out.completion_tokens or 0
        if self.on_usage:
            self.on_usage(out.prompt_tokens or 0, out.completion_tokens or 0)
        if use_cache:
            self._write_cache(key, text)
        return text

    def grade(self, prompt: str) -> dict:
        """Returns {"score": float 0..1, "reason": str}."""
        return parse_json_score(self.complete(prompt))

    def health_check(self):
        from .plugins import timed_health
        probe = 'Reply with the JSON {"ok": true}'
        return timed_health(lambda: f"{self.model}: {self.complete(probe, use_cache=False)[:40]!r}")

    def close(self):
        self.provider.close()


def build_judge(cfg: dict | None, cache_dir: str | None = None, on_usage=None) -> Judge | None:
    """Judge from a config block (None if absent)."""
    if not cfg:
        return None
    cfg = expand_env(dict(cfg))
    if "model" not in cfg:
        raise ConfigError("judge config needs 'model'")
    if cache_dir and "cache_dir" not in cfg:
        cfg["cache_dir"] = cache_dir
    return Judge(**cfg, on_usage=on_usage)


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
