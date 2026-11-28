"""PostgreSQL storage: artifacts in one table, per-case records in another
(upserted, so a re-run case replaces its old record). Needs psycopg 3:
pip install 'evalkit[postgres]'.

config:
  storage:
    type: postgres
    dsn: ${secret:evalkit-db-dsn}      # postgresql://user:pass@host:5432/evals?sslmode=require
    schema: evalkit
"""
from __future__ import annotations

import json
import re
import threading

from ..errors import ConfigError
from . import StorageBackend


class PostgresStorage(StorageBackend):
    def __init__(self, dsn: str | None = None, schema: str = "evalkit", connection=None, **_):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
            raise ConfigError(f"invalid postgres schema name {schema!r}")
        if connection is None:
            if not dsn:
                raise ConfigError("postgres storage needs 'dsn'")
            import psycopg
            connection = psycopg.connect(dsn, autocommit=True)
        self.conn, self.schema = connection, schema
        self._lock = threading.Lock()
        self._migrate()

    def _exec(self, sql: str, params=(), fetch: bool = False):
        with self._lock:
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall() if fetch else None

    def _migrate(self):
        s = self.schema
        self._exec(f"CREATE SCHEMA IF NOT EXISTS {s}")
        self._exec(f"CREATE TABLE IF NOT EXISTS {s}.objects (key TEXT PRIMARY KEY, body TEXT NOT NULL, "
                   f"updated_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        self._exec(f"CREATE TABLE IF NOT EXISTS {s}.records (run_id TEXT NOT NULL, record_key TEXT NOT NULL, "
                   f"body JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                   f"PRIMARY KEY (run_id, record_key))")

    def write_text(self, key, text):
        self._exec(f"INSERT INTO {self.schema}.objects (key, body) VALUES (%s, %s) "
                   f"ON CONFLICT (key) DO UPDATE SET body = EXCLUDED.body, updated_at = now()", (key, text))

    def read_text(self, key):
        rows = self._exec(f"SELECT body FROM {self.schema}.objects WHERE key = %s", (key,), fetch=True)
        return rows[0][0] if rows else None

    def list(self, prefix):
        rows = self._exec(f"SELECT key FROM {self.schema}.objects WHERE key LIKE %s ORDER BY key",
                          (_like(prefix),), fetch=True)
        return [r[0] for r in rows]

    def delete(self, key):
        self._exec(f"DELETE FROM {self.schema}.objects WHERE key = %s", (key,))

    def append_record(self, run_id, key, record):
        self._exec(f"INSERT INTO {self.schema}.records (run_id, record_key, body) VALUES (%s, %s, %s) "
                   f"ON CONFLICT (run_id, record_key) DO UPDATE SET body = EXCLUDED.body, created_at = now()",
                   (run_id, key, json.dumps(record, ensure_ascii=False)))

    def read_records(self, run_id):
        rows = self._exec(f"SELECT body FROM {self.schema}.records WHERE run_id = %s ORDER BY created_at",
                          (run_id,), fetch=True)
        return [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in rows]

    def list_runs(self):
        rows = self._exec(f"SELECT DISTINCT split_part(key, '/', 1) FROM {self.schema}.objects "
                          f"WHERE key LIKE '%%/%%' ORDER BY 1", fetch=True)
        return [r[0] for r in rows if not r[0].startswith("_")]

    def delete_run(self, run_id):
        n = len(self.list(f"{run_id}/"))
        self._exec(f"DELETE FROM {self.schema}.objects WHERE key LIKE %s", (_like(f"{run_id}/"),))
        self._exec(f"DELETE FROM {self.schema}.records WHERE run_id = %s", (run_id,))
        return n

    def close(self):
        self.conn.close()


def _like(prefix: str) -> str:
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
