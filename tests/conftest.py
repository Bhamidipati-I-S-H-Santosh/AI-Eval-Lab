"""Shared test setup: keep the audit log and judge caches out of the repo."""
import os
import tempfile

os.environ.setdefault("EVALKIT_AUDIT_LOG", os.path.join(tempfile.mkdtemp(prefix="evalkit-audit-"), "audit.jsonl"))
