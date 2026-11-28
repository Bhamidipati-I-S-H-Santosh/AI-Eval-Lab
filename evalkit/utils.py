from __future__ import annotations

import importlib
import os
import re
from typing import Any

_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Replace ${VAR} or ${VAR:-default} in strings (recursively), so secrets
    stay in the environment / .env, never in config files."""
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def import_object(path: str) -> Any:
    """'package.module:attribute' -> the attribute."""
    module, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attribute', got '{path}'")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def get_path(data: Any, path: str | None, default: Any = None) -> Any:
    """Read a dotted path like 'reply.text' or 'choices.0.message' from JSON."""
    if not path:
        return default
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return default
    return cur


def fill_template(template: Any, values: dict) -> Any:
    """Fill '{question}' placeholders inside a JSON request template."""
    if isinstance(template, str):
        if template.strip() in {f"{{{k}}}" for k in values}:  # keep non-string types intact
            return values[template.strip()[1:-1]]
        out = template
        for k, v in values.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(template, dict):
        return {k: fill_template(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [fill_template(v, values) for v in template]
    return template


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k successes out of n."""
    import math
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test (McNemar for paired 0/1 outcomes); ties are dropped."""
    import math
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
