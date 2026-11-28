"""Where runs live: datasets, per-case results, scores and reports.

Every run has an id and a set of keys under it:
    <run_id>/config.yaml  cases.jsonl  meta.json  report.json  report.md  scores.jsonl
    <run_id>/results.jsonl            (local)   or   <run_id>/records/<case>__r<n>.json (object stores)

Results are written one record at a time, the moment each case finishes, so an
interrupted run loses at most the cases that were in flight. Object stores
can't append, so each record is its own object; reading merges them.

config:
  storage:
    type: s3                    # local (default) | s3 | azure_blob | gcs | postgres
    bucket: my-evals
    prefix: evalkit/
"""
from __future__ import annotations

import json
from abc import abstractmethod

from ..plugins import HealthStatus, Plugin, timed_health


def record_key(case_id: str, repeat: int) -> str:
    return f"{case_id}__r{repeat}"


class StorageBackend(Plugin):
    """Key/value text storage plus durable per-result records."""

    @abstractmethod
    def write_text(self, key: str, text: str) -> None: ...

    @abstractmethod
    def read_text(self, key: str) -> str | None: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    def exists(self, key: str) -> bool:
        return self.read_text(key) is not None

    # ----- records: one per (case, repeat) -----
    def append_record(self, run_id: str, key: str, record: dict) -> None:
        self.write_text(f"{run_id}/records/{key}.json", json.dumps(record, ensure_ascii=False))

    def read_records(self, run_id: str) -> list[dict]:
        out = []
        for k in sorted(self.list(f"{run_id}/records/")):
            text = self.read_text(k)
            if text:
                try:
                    out.append(json.loads(text))
                except ValueError:
                    continue      # a torn write; that case will simply run again
        return out

    def list_runs(self) -> list[str]:
        return sorted({k.split("/", 1)[0] for k in self.list("") if "/" in k})

    def delete_run(self, run_id: str) -> int:
        keys = self.list(f"{run_id}/")
        for k in keys:
            self.delete(k)
        return len(keys)

    def health_check(self) -> HealthStatus:
        def probe():
            self.write_text("_health/probe.txt", "ok")
            assert self.read_text("_health/probe.txt") == "ok", "read-back mismatch"
            self.delete("_health/probe.txt")
            return "write/read/delete ok"
        return timed_health(probe)


def build_storage(cfg: dict | None, default_root: str = "results/evalkit") -> StorageBackend:
    from ..plugins import REGISTRY
    if not cfg:
        cfg = {"type": "local", "root": default_root}
    return REGISTRY.build("storage", cfg)


__all__ = ["StorageBackend", "build_storage", "record_key"]
