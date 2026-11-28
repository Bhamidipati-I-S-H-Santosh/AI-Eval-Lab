"""Local filesystem storage. Records are appended to <run>/results.jsonl
(the format evalkit has always used) and fsync'd, so a crash or power loss
keeps every finished case."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from ..errors import ConfigError
from . import StorageBackend


class LocalStorage(StorageBackend):
    def __init__(self, root: str = "results/evalkit", fsync: bool = True, **_):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ConfigError(f"storage key escapes the storage root: {key!r}")
        return p

    def write_text(self, key, text):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)

    def read_text(self, key):
        p = self._path(key)
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def list(self, prefix):
        base = self.root
        out = []
        for p in base.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                k = p.relative_to(base).as_posix()
                if k.startswith(prefix):
                    out.append(k)
        return sorted(out)

    def delete(self, key):
        p = self._path(key)
        if p.is_file():
            p.unlink()

    def delete_run(self, run_id):
        n = super().delete_run(run_id)
        d = self._path(run_id)
        for sub in sorted((x for x in d.rglob("*") if x.is_dir()), reverse=True) if d.exists() else []:
            try:
                sub.rmdir()
            except OSError:
                pass
        if d.exists():
            try:
                d.rmdir()
            except OSError:
                pass
        return n

    def run_dir(self, run_id: str) -> Path:
        d = self._path(run_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def append_record(self, run_id, key, record):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        path = self.run_dir(run_id) / "results.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())

    def read_records(self, run_id):
        path = self._path(run_id) / "results.jsonl"
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue   # torn last line after a crash: that case reruns
        return out

    def list_runs(self):
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("_"))
