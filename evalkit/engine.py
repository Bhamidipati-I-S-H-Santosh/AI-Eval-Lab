"""Execution controls shared by every run: rate limiting, budgets, retries.

config:
  run:
    concurrency: 4
    rate_limit: {requests_per_minute: 30, tokens_per_minute: 8000}
    retries: {max_attempts: 3, backoff_s: 5, max_backoff_s: 60}
  budget:
    max_tokens: 2000000        # system + judge tokens for this run
    max_cost_usd: 5.00         # needs cost.input_per_1m / output_per_1m
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass

from .errors import BudgetExceededError

log = logging.getLogger("evalkit.engine")


class RateLimiter:
    """Token bucket on requests per minute and (after the fact) tokens per
    minute. Thread-safe; callers block in acquire() until they may go."""

    def __init__(self, requests_per_minute: float | None = None, tokens_per_minute: float | None = None,
                 clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm = requests_per_minute, tokens_per_minute
        self.clock, self.sleep = clock, sleep
        self._lock = threading.Lock()
        now = clock()
        self._req_tokens, self._req_t = float(requests_per_minute or 0), now
        self._tok_tokens, self._tok_t = float(tokens_per_minute or 0), now

    @property
    def active(self) -> bool:
        return bool(self.rpm or self.tpm)

    def _refill(self, now):
        if self.rpm:
            self._req_tokens = min(self.rpm, self._req_tokens + (now - self._req_t) * self.rpm / 60)
            self._req_t = now
        if self.tpm:
            self._tok_tokens = min(self.tpm, self._tok_tokens + (now - self._tok_t) * self.tpm / 60)
            self._tok_t = now

    def acquire(self) -> float:
        """Wait for a request slot. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self.clock()
                self._refill(now)
                wait = 0.0
                if self.rpm and self._req_tokens < 1:
                    wait = max(wait, (1 - self._req_tokens) * 60 / self.rpm)
                if self.tpm and self._tok_tokens < 0:
                    wait = max(wait, -self._tok_tokens * 60 / self.tpm)
                if wait <= 0:
                    if self.rpm:
                        self._req_tokens -= 1
                    return waited
            self.sleep(min(wait, 5.0))
            waited += min(wait, 5.0)

    def record_tokens(self, n: int) -> None:
        if self.tpm and n:
            with self._lock:
                self._refill(self.clock())
                self._tok_tokens -= n


@dataclass
class Budget:
    max_tokens: float | None = None
    max_cost_usd: float | None = None
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0

    def __post_init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = self.completion_tokens = 0

    def charge(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt or 0
            self.completion_tokens += completion or 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * self.price_in_per_1m + self.completion_tokens * self.price_out_per_1m) / 1e6

    def exceeded(self) -> str | None:
        if self.max_tokens and self.tokens >= self.max_tokens:
            return f"token budget reached ({self.tokens} >= {int(self.max_tokens)})"
        if self.max_cost_usd and self.cost_usd >= self.max_cost_usd:
            return f"cost budget reached (${self.cost_usd:.4f} >= ${self.max_cost_usd})"
        return None

    def check(self) -> None:
        reason = self.exceeded()
        if reason:
            raise BudgetExceededError(reason)

    def to_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6), "max_tokens": self.max_tokens,
                "max_cost_usd": self.max_cost_usd}


_TRANSIENT = re.compile(r"\b(429|500|502|503|504|rate.?limit|too many requests|timed? ?out|timeout|"
                        r"temporarily|overloaded|connection (reset|aborted|error)|service unavailable)\b", re.I)


@dataclass
class RetryPolicy:
    max_attempts: int = 1           # 1 = no retry (the system's own client usually retries)
    backoff_s: float = 5.0
    max_backoff_s: float = 60.0
    retry_on: str = "transient"     # transient | any

    def should_retry(self, error: str | None, attempt: int) -> bool:
        if not error or attempt >= self.max_attempts:
            return False
        return self.retry_on == "any" or bool(_TRANSIENT.search(error))

    def delay(self, attempt: int) -> float:
        return min(self.max_backoff_s, self.backoff_s * 2 ** (attempt - 1))


def from_config(cfg: dict) -> tuple[RateLimiter, Budget, RetryPolicy]:
    run = cfg.get("run") or {}
    rl = run.get("rate_limit") or {}
    rt = run.get("retries")
    retry = RetryPolicy(**rt) if isinstance(rt, dict) else RetryPolicy(max_attempts=int(rt or 0) + 1)
    b = cfg.get("budget") or {}
    cost = cfg.get("cost") or {}
    budget = Budget(max_tokens=b.get("max_tokens"), max_cost_usd=b.get("max_cost_usd"),
                    price_in_per_1m=cost.get("input_per_1m", 0.0), price_out_per_1m=cost.get("output_per_1m", 0.0))
    return RateLimiter(rl.get("requests_per_minute"), rl.get("tokens_per_minute")), budget, retry
