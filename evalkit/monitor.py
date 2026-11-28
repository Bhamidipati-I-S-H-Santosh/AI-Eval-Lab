"""Production monitoring: score sampled live traffic, track trends, alert on regressions.

Live agents export OpenTelemetry traces to evalkit's receiver (POST /v1/traces)
or to storage directly. `python -m evalkit monitor CONFIG` then:

  1. loads the traces recorded in the window (default: last 24 h),
  2. samples them (sample_rate) and normalizes them (evalkit.otel),
  3. scores them with reference-free metrics: error, latency, tokens, steps,
     faithfulness and relevance (judge), gave_up (the judge's abstention check),
  4. breaks results down by agent version (resource attribute service.version),
  5. appends the window to monitoring/history.jsonl and compares it with the
     previous window: a drop that is statistically significant (p < alpha) and
     larger than min_delta raises an alert (optionally POSTed to a webhook),
  6. lists the most frequent failure patterns (error types, failing tools).

config:
  monitoring:
    traces_prefix: otel/
    window_hours: 24
    sample_rate: 0.2
    min_n: 20                 # fewer scored traces than this: no alerting
    alpha: 0.05
    min_delta: 0.05           # ignore significant but tiny changes
    webhook: https://hooks.example/...   # optional; subject to privacy.allowed_hosts
"""
from __future__ import annotations

import json
import math
import random
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import metrics as M
from .otel import normalize_spans, parse_otlp
from .types import TestCase

MONITOR_METRICS = ["error", "latency_s", "total_tokens", "steps", "faithfulness", "relevance", "gave_up"]


@M.metric("gave_up", higher_is_better=False)
def m_gave_up(case, r, ctx):
    """Monitoring only: share of live answers that gave up ('I don't know')."""
    if not case.metadata.get("monitoring") or r.error:
        return None
    return float(M._gave_up(case, r, ctx))


# ---------- ingestion ----------
def ingest(storage, payload, prefix: str = "otel/") -> dict:
    """Validate and store one OTLP export request. Returns counts."""
    spans = parse_otlp(payload)
    if not spans:
        raise ValueError("no spans with trace and span ids in the payload")
    now = datetime.now(timezone.utc)
    key = f"{prefix}{now:%Y/%m/%d/%H%M%S}-{uuid.uuid4().hex[:12]}.json"
    if isinstance(payload, (bytes, bytearray)):
        try:
            body = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            from .otel.normalize import _protobuf_to_dict
            body = _protobuf_to_dict(bytes(payload))
    else:
        body = json.loads(payload) if isinstance(payload, str) else payload
    storage.write_text(key, json.dumps(body, ensure_ascii=False))
    return {"stored": key, "spans": len(spans), "traces": len({s.trace_id for s in spans})}


def _window_keys(storage, prefix: str, start: datetime, end: datetime) -> list[str]:
    keys = []
    day = start.date()
    while day <= end.date():
        keys += storage.list(f"{prefix}{day:%Y/%m/%d}/")
        day += timedelta(days=1)
    return keys


# ---------- statistics ----------
def _two_prop_p(k1, n1, k2, n2) -> float:
    if not n1 or not n2:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    if se == 0:
        return 1.0 if k1 / n1 == k2 / n2 else 0.0
    z = abs(k1 / n1 - k2 / n2) / se
    return math.erfc(z / math.sqrt(2))


def _welch_p(m1, s1, n1, m2, s2, n2) -> float:
    if n1 < 2 or n2 < 2:
        return 1.0
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    if se == 0:
        return 1.0 if m1 == m2 else 0.0
    return math.erfc(abs(m1 - m2) / se / math.sqrt(2))      # normal approximation


def _stats(values: list[float]) -> dict:
    n = len(values)
    if not n:
        return {"n": 0}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    binary = all(v in (0.0, 1.0) for v in values)
    return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4), "k": int(sum(values)) if binary else None}


def compare_windows(prev: dict, cur: dict, alpha: float, min_delta: float, min_n: int) -> list[dict]:
    """Alerts for metrics that got significantly worse between two windows."""
    alerts = []
    for m, c in cur.items():
        p = prev.get(m)
        if not p or not c.get("n") or not p.get("n") or c["n"] < min_n or p["n"] < min_n:
            continue
        hib = M.higher_is_better(m) if m in M.REGISTRY else True
        worse = (c["mean"] < p["mean"]) if hib else (c["mean"] > p["mean"])
        rel = abs(c["mean"] - p["mean"]) / (abs(p["mean"]) or 1.0) if m in ("latency_s", "total_tokens", "steps") \
            else abs(c["mean"] - p["mean"])
        if not worse or rel < min_delta:
            continue
        pv = _two_prop_p(c["k"], c["n"], p["k"], p["n"]) if c.get("k") is not None and p.get("k") is not None \
            else _welch_p(c["mean"], c["sd"], c["n"], p["mean"], p["sd"], p["n"])
        if pv < alpha:
            alerts.append({"metric": m, "previous": p["mean"], "current": c["mean"], "p_value": round(pv, 5),
                           "n_previous": p["n"], "n_current": c["n"]})
    return alerts


def failure_patterns(results: list, top: int = 10) -> dict:
    errors = Counter()
    tools = Counter()
    for r in results:
        if r.error:
            errors[re.sub(r"\d+", "N", r.error)[:120]] += 1
        for t in r.trace:
            if re.search(r'"?error"?\s*[:=]|exception|traceback|failed', t.result or "", re.I):
                tools[t.tool] += 1
    return {"errors": errors.most_common(top), "failing_tools": tools.most_common(top)}


# ---------- the monitoring run ----------
def monitor(config_path: str, window_hours: float | None = None, now: datetime | None = None, log=print) -> dict:
    from . import audit
    from .judge import build_judge
    from .runner import _storages, aggregate, load_config, score_rows
    cfg = load_config(config_path)
    mcfg = cfg.get("monitoring") or {}
    local, mirror = _storages(cfg)
    store = mirror or local
    prefix = mcfg.get("traces_prefix", "otel/")
    hours = window_hours or mcfg.get("window_hours", 24)
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    spans = []
    for k in _window_keys(store, prefix, start, end):
        text = store.read_text(k)
        if text:
            try:
                spans += parse_otlp(text)
            except ValueError:
                log(f"  ! skipping unreadable trace file {k}")
    traces = [t for t in normalize_spans(spans)
              if t.start_ns is None or start.timestamp() <= t.start_ns / 1e9 <= end.timestamp()]
    rate = float(mcfg.get("sample_rate", 1.0))
    rng = random.Random(mcfg.get("seed", int(end.timestamp())))
    sampled = [t for t in traces if rng.random() < rate]
    log(f"Monitoring window {start:%Y-%m-%d %H:%M} .. {end:%H:%M} UTC: {len(traces)} trace(s), {len(sampled)} sampled")

    cases, rows = [], []
    for t in sampled:
        ver = t.service_version or "unknown"
        cases.append(TestCase(id=f"trace-{t.trace_id[:16]}", input=t.question or "",
                              tags=[f"version:{ver}"] + ([f"agent:{t.agent_name}"] if t.agent_name else []),
                              metadata={"monitoring": True, "trace_id": t.trace_id, "model": t.model}))
        rows.append({"case_id": cases[-1].id, "repeat": 0, "result": t.result.to_dict()})
    judge = build_judge(cfg.get("judge"))
    score_cfg = {**cfg, "metrics": [m for m in mcfg.get("metrics", MONITOR_METRICS) if m in M.REGISTRY]}
    scored = score_rows(rows, cases, score_cfg, judge) if rows else []
    summary = aggregate(scored, cases) if scored else {"overall": {}, "by_tag": {}, "flaky_cases": []}

    def window_stats(rs):
        out = {}
        for m in score_cfg["metrics"]:
            vals = [r["scores"].get(m) for r in rs if r["scores"].get(m) is not None]
            if vals:
                out[m] = _stats(vals)
        return out

    by_case = {c.id: c for c in cases}
    versions = sorted({t.split(":", 1)[1] for c in cases for t in c.tags if t.startswith("version:")})
    current = {"window_end": end.isoformat(timespec="seconds"), "window_hours": hours, "n": len(scored),
               "overall": window_stats(scored),
               "by_version": {v: window_stats([r for r in scored if f"version:{v}" in by_case[r["case_id"]].tags])
                              for v in versions}}

    hist_key = "monitoring/history.jsonl"
    history_text = store.read_text(hist_key) or ""
    history = [json.loads(l) for l in history_text.splitlines() if l.strip()]
    alpha, min_delta, min_n = mcfg.get("alpha", 0.05), mcfg.get("min_delta", 0.05), mcfg.get("min_n", 20)
    alerts = []
    if history:
        prev = history[-1]
        alerts = [{"scope": "overall", **a} for a in compare_windows(prev["overall"], current["overall"],
                                                                      alpha, min_delta, min_n)]
    if len(versions) > 1:          # a new version deployed inside this window: compare it with the previous one
        new, old = versions[-1], versions[-2]
        alerts += [{"scope": f"version {old} -> {new}", **a} for a in compare_windows(
            current["by_version"][old], current["by_version"][new], alpha, min_delta, min_n)]
    current["alerts"] = alerts
    current["failure_patterns"] = failure_patterns([AgentResultView(r) for r in rows])
    store.write_text(hist_key, history_text + json.dumps(current) + "\n")

    run_dir = Path(local.run_dir(f"monitor-{end:%Y%m%d-%H%M%S}"))
    (run_dir / "monitor.json").write_text(json.dumps({**current, "summary": summary}, indent=2), encoding="utf-8")
    _write_md(run_dir / "report.md", current, cfg.get("name", "monitoring"))
    if alerts and mcfg.get("webhook"):
        _post_webhook(mcfg["webhook"], cfg.get("name", "evalkit"), alerts, log)
    audit.emit("monitor.window", traces=len(traces), scored=len(scored), alerts=len(alerts))
    for a in alerts:
        log(f"  ALERT [{a['scope']}] {a['metric']}: {a['previous']} -> {a['current']} (p={a['p_value']})")
    return {**current, "run_dir": str(run_dir)}


class AgentResultView:
    """Light view over a stored result dict for failure_patterns."""

    def __init__(self, row):
        from .types import AgentResult
        self._r = AgentResult.from_dict(row["result"])

    def __getattr__(self, k):
        return getattr(self._r, k)


def _write_md(path: Path, cur: dict, name: str):
    lines = [f"# Monitoring: {name}", "", f"- Window end: {cur['window_end']} ({cur['window_hours']} h)",
             f"- Scored traces: {cur['n']}", f"- Alerts: {len(cur['alerts'])}", "",
             "| Metric | Mean | n |", "|---|---|---|"]
    for m, s in cur["overall"].items():
        lines.append(f"| {m} | {s.get('mean')} | {s.get('n')} |")
    if cur["by_version"]:
        lines += ["", "## By agent version", ""]
        for v, st in cur["by_version"].items():
            lines.append(f"- **{v}**: " + ", ".join(f"{m}={s.get('mean')} (n={s.get('n')})" for m, s in st.items()))
    if cur["alerts"]:
        lines += ["", "## Alerts (significant regressions)", "", "| Scope | Metric | Before | Now | p |",
                  "|---|---|---|---|---|"]
        for a in cur["alerts"]:
            lines.append(f"| {a['scope']} | {a['metric']} | {a['previous']} | {a['current']} | {a['p_value']} |")
    fp = cur["failure_patterns"]
    if fp["errors"] or fp["failing_tools"]:
        lines += ["", "## Failure patterns", ""]
        lines += [f"- {n}x error: `{e}`" for e, n in fp["errors"]]
        lines += [f"- {n}x failing tool: `{t}`" for t, n in fp["failing_tools"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _post_webhook(url: str, name: str, alerts: list, log):
    import requests
    text = f"evalkit monitoring '{name}': {len(alerts)} regression(s): " + "; ".join(
        f"{a['metric']} {a['previous']}->{a['current']} ({a['scope']})" for a in alerts)
    try:
        requests.post(url, json={"text": text, "alerts": alerts}, timeout=10)
    except Exception as e:
        log(f"  ! webhook failed: {type(e).__name__}")
