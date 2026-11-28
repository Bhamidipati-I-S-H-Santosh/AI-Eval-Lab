"""Privacy controls: PII redaction and a data-egress allowlist.

config:
  privacy:
    redact_reports: true          # mask PII in report.md (inputs/answers shown there)
    redact_stored: false          # mask PII in results before they are written (scores
                                  # are computed afterwards, so correctness checks see masked text)
    detectors: [email, phone, credit_card, ssn, iban, ip, credential]   # default: all
    allowed_hosts:                # data egress allowlist (opt-in, enforced at startup):
      - "*.openai.azure.com"      # every endpoint the run will send data to must match,
      - "bot.corp.example"        # including implicit vendor endpoints (api.openai.com, ...)
      - localhost

Secrets resolved through evalkit.secrets are always masked in logs and reports.
evalkit itself sends no telemetry anywhere; data only goes to the endpoints
named in your config (see `python -m evalkit validate` for the egress summary).
"""
from __future__ import annotations

import fnmatch
import logging
import re
from urllib.parse import urlparse

from .errors import ConfigError

DETECTORS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b"),
    # not part of a longer digit run (card-like numbers that fail Luhn, ids)
    "phone": re.compile(r"(?<![\w$])(?<!\d[ .-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]\d{3,4}"
                        r"(?:[ .-]\d{2,4})?(?![\w%])(?![ .-]\d)"),
    "ip": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "credential": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                             r"ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"),
}
ORDER = ["credential", "email", "iban", "credit_card", "ssn", "ip", "phone"]


def _luhn(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(d)):
        if i % 2:
            n *= 2
            n -= 9 if n > 9 else 0
        total += n
    return total % 10 == 0


def redact_pii(text: str, detectors: list[str] | None = None) -> tuple[str, dict[str, int]]:
    """Mask PII. Returns (text, {detector: count})."""
    if not text:
        return text, {}
    names = [n for n in ORDER if detectors is None or n in detectors]
    counts: dict[str, int] = {}
    for name in names:
        pat = DETECTORS[name]

        def sub(m, name=name):
            if name == "credit_card" and not _luhn(m.group(0)):
                return m.group(0)
            counts[name] = counts.get(name, 0) + 1
            return f"[{name.upper()}]"
        text = pat.sub(sub, text)
    return text, counts


class Redactor:
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.reports = bool(cfg.get("redact_reports", False))
        self.stored = bool(cfg.get("redact_stored", False))
        self.detectors = cfg.get("detectors")
        unknown = set(self.detectors or []) - set(DETECTORS)
        if unknown:
            raise ConfigError(f"unknown privacy detectors {sorted(unknown)}; known: {sorted(DETECTORS)}")

    def text(self, s: str) -> str:
        from .secrets import redact as redact_secrets
        s = redact_secrets(s)
        return redact_pii(s, self.detectors)[0]

    def result(self, result: dict) -> dict:
        """Mask a stored AgentResult dict (answer, tool args/results, retrieved text)."""
        r = dict(result)
        r["answer"] = self.text(r.get("answer") or "")
        r["trace"] = [{**t, "args": _redact_obj(t.get("args"), self), "result": self.text(t.get("result") or "")}
                      for t in r.get("trace") or []]
        r["retrieved"] = [{**d, "text": self.text(d.get("text") or "")} for d in r.get("retrieved") or []]
        if r.get("raw"):
            r["raw"] = self.text(str(r["raw"]))
        return r


def _redact_obj(o, red: Redactor):
    if isinstance(o, str):
        return red.text(o)
    if isinstance(o, dict):
        return {k: _redact_obj(v, red) for k, v in o.items()}
    if isinstance(o, list):
        return [_redact_obj(v, red) for v in o]
    return o


class RedactingFilter(logging.Filter):
    """Masks secrets and PII in every evalkit log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean, _ = redact_pii(msg)
        from .secrets import redact as redact_secrets
        clean = redact_secrets(clean)
        if clean != msg:
            record.msg, record.args = clean, None
        return True


def install_log_redaction() -> None:
    logger = logging.getLogger("evalkit")
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())


# ---------- data egress ----------
_URL_KEYS = re.compile(r"(url|endpoint|base_url|host|azure_endpoint|account_url|token_url|graph_url|webhook)$", re.I)
_IMPLICIT = {  # vendor endpoints used when no URL is configured
    ("model", "openai_compatible"): "api.openai.com",
    ("model", "anthropic"): "api.anthropic.com",
    ("model", "bedrock"): "bedrock-runtime.*.amazonaws.com",
    ("model", "vertex"): "*-aiplatform.googleapis.com",
    ("model", "ollama"): "localhost",
    ("system", "openai_responses"): "api.openai.com",
    ("system", "openai_assistants"): "api.openai.com",
    ("system", "bedrock_agent"): "bedrock-agent-runtime.*.amazonaws.com",
    ("system", "copilot_studio"): "directline.botframework.com",
    ("storage", "s3"): "s3.*.amazonaws.com",
    ("storage", "gcs"): "storage.googleapis.com",
    ("secrets", "aws_secrets_manager"): "secretsmanager.*.amazonaws.com",
    ("secrets", "gcp_secret_manager"): "secretmanager.googleapis.com",
    ("identity", "sharepoint"): "graph.microsoft.com",
    ("identity", "google_drive"): "www.googleapis.com",
}


def _hosts_in(obj, out: set):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and _URL_KEYS.search(str(k)) and v:
                h = urlparse(v).hostname if "://" in v else v.split("/")[0].split(":")[0]
                if h:
                    out.add(h.lower())
            else:
                _hosts_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _hosts_in(v, out)


def egress_hosts(cfg: dict) -> list[tuple[str, str]]:
    """(component, host) for every destination the config will send data to."""
    from .secrets import resolve_config
    found: list[tuple[str, str]] = []
    blocks = {"system": cfg.get("system"), "retriever": cfg.get("retriever"), "storage": cfg.get("storage"),
              "secrets": cfg.get("secrets"), "identity": (cfg.get("identity") or {}).get("acl"),
              "monitoring": cfg.get("monitoring")}
    judge = cfg.get("judge")
    if judge:
        prov = judge.get("provider") or "openai_compatible"
        blocks["judge"] = {**judge, "type": prov if isinstance(prov, str) else prov.get("type")}
    for comp, block in blocks.items():
        if not isinstance(block, dict):
            continue
        try:
            block = resolve_config(block)
        except Exception:
            pass
        hosts: set = set()
        _hosts_in(block, hosts)
        if not hosts:
            cat = {"judge": "model"}.get(comp, comp)
            implicit = _IMPLICIT.get((cat, block.get("type")))
            if implicit:
                hosts.add(implicit)
        found += [(comp, h) for h in sorted(hosts)]
    return found


def check_egress(cfg: dict) -> list[tuple[str, str]]:
    """Enforce privacy.allowed_hosts (if set). Returns the egress list."""
    hosts = egress_hosts(cfg)
    allowed = ((cfg.get("privacy") or {}).get("allowed_hosts"))
    if allowed is None:
        return hosts
    bad = [(c, h) for c, h in hosts if not any(fnmatch.fnmatch(h, pat.lower()) for pat in allowed)]
    if bad:
        raise ConfigError("data egress blocked by privacy.allowed_hosts: "
                          + ", ".join(f"{c} -> {h}" for c, h in bad)
                          + ". Add the host to the allowlist only if sending evaluation data there is approved.")
    return hosts
