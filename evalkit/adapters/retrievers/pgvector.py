"""Connect a PostgreSQL table that stores embeddings with pgvector.

config:
  type: pgvector
  dsn: ${PG_DSN}                      # postgresql://user:pass@host:5432/db
  table: public.chunks                # optionally schema-qualified
  text_column: content
  id_column: id
  embedding_column: embedding
  metadata_columns: [title, url]      # returned in doc.metadata
  source_column: url                  # optional: copied to metadata["source"]
  distance: cosine                    # cosine (<=>) | l2 (<->) | inner_product (<#>)
  filter: { tenant: acme }            # optional fixed WHERE col = value (parameterized)
  embedder: sentence-transformers:all-MiniLM-L6-v2   # must match how the index was built

Identifiers (table/column names) are validated against a strict regex and
double-quoted, so they are case-sensitive: a table created unquoted as Chunks
is stored as `chunks`. Values (query vector, filter values, k) are always sent
as bind parameters, never interpolated.

score = similarity, higher is better: cosine -> 1 - distance,
inner_product -> -(<#> result) = inner product, l2 -> -distance.
"""
from __future__ import annotations

import re

from ...embeddings import get_embedder
from ...errors import ConfigError, PluginDependencyError
from ...plugins import HealthStatus, timed_health
from ...types import RetrievedDoc
from ...utils import expand_env
from ..base import RetrieverAdapter

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_OPS = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}


def quote_ident(name: str, what: str = "identifier", allow_schema: bool = False) -> str:
    """Validate and double-quote a SQL identifier. Raises ConfigError on
    anything that is not a plain name (or schema.name when allowed)."""
    if not isinstance(name, str):
        raise ConfigError(f"pgvector: {what} must be a string, got {type(name).__name__}")
    parts = name.split(".") if allow_schema else [name]
    if len(parts) > 2 or not all(_IDENT.match(p) for p in parts):
        raise ConfigError(f"pgvector: invalid {what} {name!r}. Use letters, digits and underscores "
                          f"only (max 63 chars, not starting with a digit)"
                          + ("; optionally schema-qualified as schema.table" if allow_schema else ""))
    return ".".join(f'"{p}"' for p in parts)


def _vector_literal(vec) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class PgVectorRetriever(RetrieverAdapter):
    def __init__(self, table: str, dsn: str | None = None, text_column: str = "content",
                 id_column: str = "id", embedding_column: str = "embedding",
                 metadata_columns: list[str] | None = None, source_column: str | None = None,
                 distance: str = "cosine", filter: dict | None = None,
                 embedder: str = "sentence-transformers:all-MiniLM-L6-v2",
                 connection=None, **_):
        if distance not in _OPS:
            raise ConfigError(f"pgvector: distance must be one of {sorted(_OPS)}, got {distance!r}")
        self.distance, self.op = distance, _OPS[distance]
        self.table = quote_ident(table, "table", allow_schema=True)
        self.text_col = quote_ident(text_column, "text_column")
        self.id_col = quote_ident(id_column, "id_column")
        self.emb_col = quote_ident(embedding_column, "embedding_column")
        self.meta_names = list(metadata_columns or [])
        self.meta_cols = [quote_ident(c, "metadata column") for c in self.meta_names]
        self.source_name = source_column
        self.source_col = quote_ident(source_column, "source_column") if source_column else None
        self.filter_sql, self.filter_params = self._build_filter(filter)

        self._owns = connection is None
        if connection is None:
            if not dsn:
                raise ConfigError("pgvector: set 'dsn' (e.g. postgresql://user:pass@host:5432/db) "
                                  "or inject 'connection'")
            try:
                import psycopg
            except ImportError as e:
                raise PluginDependencyError("retriever:pgvector", "psycopg", "pgvector") from e
            connection = psycopg.connect(expand_env(dsn), autocommit=True)
        self.conn = connection
        self.embed = get_embedder(embedder)

    @staticmethod
    def _build_filter(f: dict | None) -> tuple[str, list]:
        if not f:
            return "", []
        clauses, params = [], []
        for col, val in f.items():
            q = quote_ident(col, "filter column")
            if val is None:
                clauses.append(f"{q} IS NULL")
            elif isinstance(val, (list, tuple, set)):
                clauses.append(f"{q} = ANY(%s)")
                params.append(list(val))
            else:
                clauses.append(f"{q} = %s")
                params.append(val)
        return " WHERE " + " AND ".join(clauses), params

    def _select_cols(self) -> str:
        cols = [self.id_col, self.text_col] + self.meta_cols
        if self.source_col:
            cols.append(self.source_col)
        return ", ".join(cols)

    def _query(self, sql: str, params: list) -> list[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def _to_doc(self, row, score=None) -> RetrievedDoc:
        doc_id, text, *rest = row
        meta = dict(zip(self.meta_names, rest[:len(self.meta_names)]))
        if self.source_col:
            src = rest[len(self.meta_names)]
            if src is not None:
                meta["source"] = src
        return RetrievedDoc(id=str(doc_id), text="" if text is None else str(text),
                            score=score, metadata=meta)

    def _score(self, d) -> float | None:
        if d is None:
            return None
        d = float(d)
        return 1.0 - d if self.distance == "cosine" else -d

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        vec = _vector_literal(self.embed([query])[0])
        sql = (f"SELECT {self._select_cols()}, {self.emb_col} {self.op} %s::vector AS _distance "
               f"FROM {self.table}{self.filter_sql} ORDER BY _distance LIMIT %s")
        rows = self._query(sql, [vec, *self.filter_params, int(k)])
        return [self._to_doc(r[:-1], self._score(r[-1])) for r in rows]

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        # setseed() makes random() reproducible within this session. ORDER BY
        # random() scans the table; fine for evaluation-sized corpora.
        with self.conn.cursor() as cur:
            cur.execute("SELECT setseed(%s)", [(seed % 1_000_003) / 1_000_003])
            cur.execute(f"SELECT {self._select_cols()} FROM {self.table}{self.filter_sql} "
                        f"ORDER BY random() LIMIT %s", [*self.filter_params, int(n)])
            rows = list(cur.fetchall())
        return [self._to_doc(r) for r in rows]

    def health_check(self) -> HealthStatus:
        def probe():
            rows = self._query(f"SELECT {self.id_col}, {self.text_col}, {self.emb_col} "
                               f"FROM {self.table} LIMIT 1", [])
            return f"table reachable ({'has rows' if rows else 'empty'})"
        return timed_health(probe)

    def close(self) -> None:
        if self._owns:
            self.conn.close()
