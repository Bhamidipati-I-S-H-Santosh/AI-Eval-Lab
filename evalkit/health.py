"""Health checks: build every component a config names and probe it."""
from __future__ import annotations

from .plugins import HealthStatus, timed_health


def build_components(cfg: dict, close: bool = False) -> dict:
    """Build system, retriever, judge and storage from a loaded config.
    Raises a ConfigError / PluginDependencyError with an actionable message."""
    from .runner import _storages, build_from_config
    system, retriever, judge = build_from_config(cfg)
    local, mirror = _storages(cfg)
    parts = {"system": system, "retriever": retriever, "judge": judge, "storage": mirror or local}
    if close:
        for p in parts.values():
            if p is not None and hasattr(p, "close"):
                try:
                    p.close()
                except Exception:
                    pass
    return parts


def check_config(config_path: str) -> dict[str, HealthStatus]:
    from .runner import load_config
    cfg = load_config(config_path)
    try:
        parts = build_components(cfg)
    except Exception as e:
        return {"config": HealthStatus(False, f"{type(e).__name__}: {e}")}
    out = {"config": HealthStatus(True, "valid")}
    for name, part in parts.items():
        if part is None:
            continue
        check = getattr(part, "health_check", None)
        out[name] = check() if check else timed_health(lambda: "no health check")
        if hasattr(part, "close"):
            try:
                part.close()
            except Exception:
                pass
    return out
