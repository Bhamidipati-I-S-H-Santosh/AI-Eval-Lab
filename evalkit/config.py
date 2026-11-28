"""Config validation: catch typos and wrong types before a long run starts.

Plugin blocks (system, retriever, storage, ...) are validated by the plugin
itself when it is built; this checks the overall shape and the engine's own
settings.
"""
from __future__ import annotations

import difflib

from .errors import ConfigError

TOP_LEVEL = {
    "name", "system", "retriever", "dataset", "judge", "abstention", "attacks", "run", "cost", "budget",
    "gates", "metrics", "custom_metrics", "output_dir", "storage", "secrets", "privacy", "identity",
    "audit", "monitoring", "otel", "comparison", "description",
}
RUN_KEYS = {"repeats", "concurrency", "judge_concurrency", "k", "rate_limit", "retries"}
_TYPES = {"system": dict, "retriever": dict, "dataset": dict, "judge": dict, "abstention": dict,
          "attacks": dict, "run": dict, "cost": dict, "budget": dict, "gates": dict, "metrics": list,
          "custom_metrics": list, "storage": dict, "secrets": dict, "privacy": dict, "identity": dict,
          "audit": dict, "monitoring": dict, "otel": dict, "output_dir": str, "name": str}


def _suggest(key: str, options) -> str:
    close = difflib.get_close_matches(key, sorted(options), n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


def validate_config(cfg: dict) -> None:
    errors = []
    for key, val in cfg.items():
        if key.startswith("_"):
            continue
        if key not in TOP_LEVEL:
            errors.append(f"unknown top-level key '{key}'{_suggest(key, TOP_LEVEL)}")
            continue
        want = _TYPES.get(key)
        if want and val is not None and not isinstance(val, want):
            errors.append(f"'{key}' must be a {want.__name__}, got {type(val).__name__}")
    run = cfg.get("run") or {}
    if isinstance(run, dict):
        for k in run:
            if k not in RUN_KEYS:
                errors.append(f"unknown key run.{k}{_suggest(k, RUN_KEYS)}")
        for k in ("repeats", "concurrency", "judge_concurrency", "k"):
            if k in run and (not isinstance(run[k], int) or run[k] < 1):
                errors.append(f"run.{k} must be a positive integer")
    for block in ("system", "retriever", "storage"):
        b = cfg.get(block)
        if isinstance(b, dict) and "type" not in b:
            errors.append(f"'{block}' needs a 'type'")
    gates = cfg.get("gates") or {}
    if isinstance(gates, dict):
        for m, rule in gates.items():
            r = str(rule).strip()
            num = r[2:] if r[:2] in (">=", "<=") else r
            try:
                float(num)
            except ValueError:
                errors.append(f"gate '{m}: {rule}' must look like '>=0.8', '<=0.02' or 0.8")
    budget = cfg.get("budget") or {}
    for k in budget:
        if k not in ("max_tokens", "max_cost_usd"):
            errors.append(f"unknown key budget.{k}{_suggest(k, ['max_tokens', 'max_cost_usd'])}")
    if errors:
        raise ConfigError("invalid config " + str(cfg.get("_path", "")) + ":\n  - " + "\n  - ".join(errors))
