"""Append-only audit log: who did what, when.

Each line is a JSON event chained to the previous one by hash, so any edit or
deletion in the middle of the file is detectable (python -m evalkit audit-verify).
Events never contain answers, documents or secrets: only ids, counts and outcomes.

Location: EVALKIT_AUDIT_LOG, or <output_dir>/_audit/audit.jsonl by default.
Actor: EVALKIT_ACTOR (the API sets it per request from the signed-in user),
otherwise the OS user.
"""
from __future__ import annotations

import contextvars
import getpass
import hashlib
import json
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_actor: contextvars.ContextVar[str | None] = contextvars.ContextVar("evalkit_actor", default=None)
_FORBIDDEN = {"answer", "text", "content", "prompt", "api_key", "token", "password", "secret"}


def default_path() -> Path:
    return Path(os.environ.get("EVALKIT_AUDIT_LOG", "results/evalkit/_audit/audit.jsonl"))


def set_actor(actor: str | None):
    """Set the acting user for this thread/request (returns a reset token)."""
    return _actor.set(actor)


def current_actor() -> str:
    a = _actor.get() or os.environ.get("EVALKIT_ACTOR")
    if a:
        return a
    try:
        return f"{getpass.getuser()}@{socket.gethostname()}"
    except Exception:
        return "unknown"


def _last_hash(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "0" * 64
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 4096))
        lines = f.read().decode("utf-8", "replace").strip().splitlines()
    try:
        return json.loads(lines[-1])["hash"]
    except (ValueError, KeyError, IndexError):
        return "0" * 64


def emit(event: str, path: str | Path | None = None, **fields) -> dict | None:
    """Record an event. Never raises: auditing must not break a run."""
    if os.environ.get("EVALKIT_AUDIT", "on").lower() == "off":
        return None
    path = Path(path) if path else default_path()
    clean = {k: v for k, v in fields.items() if k.lower() not in _FORBIDDEN}
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            prev = _last_hash(path)
            entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                     "actor": current_actor(), "event": event, **clean, "prev": prev}
            body = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
            entry["hash"] = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return entry
    except Exception:
        return None


def verify(path: str | Path | None = None) -> tuple[bool, str]:
    """Check the hash chain. Returns (ok, message)."""
    path = Path(path) if path else default_path()
    if not path.exists():
        return True, "no audit log yet"
    prev = "0" * 64
    count = 0
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        entry = json.loads(line)
        h = entry.pop("hash", None)
        if entry.get("prev") != prev:
            return False, f"line {n}: chain broken (an earlier line was edited or removed)"
        body = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
        if hashlib.sha256((prev + body).encode("utf-8")).hexdigest() != h:
            return False, f"line {n}: contents were modified"
        prev = h
    return True, f"{count} event(s), chain intact"
