"""Connect an Azure AI Search index over its REST API (no SDK).

config:
  type: azure_ai_search
  endpoint: https://my-search.search.windows.net
  index: company-docs
  api_key: ${AZURE_SEARCH_KEY}        # query or admin key, sent as the api-key header
  # or Entra ID (needs azure-identity; scope defaults to https://search.azure.com/.default):
  # auth: { type: entra_id }
  mode: hybrid                        # keyword | vector | hybrid | semantic
  vector_fields: [contentVector]      # vector / hybrid (and optional for semantic)
  embedder: sentence-transformers:all-MiniLM-L6-v2   # local embedding -> vectorQueries kind "vector"
                                      # omit to use integrated vectorization (kind "text")
  semantic_configuration: default     # semantic mode
  text_field: content
  id_field: id
  source_field: metadata_storage_name # copied to metadata["source"] if present
  select: [id, content, title, metadata_storage_name]   # optional $select
  search_fields: [content, title]     # optional searchFields (keyword part)
  filter: { category: finance }       # dict -> OData 'eq' clauses, or a raw OData string
  api_version: 2024-07-01

score: @search.rerankerScore in semantic mode, otherwise @search.score.
"""
from __future__ import annotations

import random
import re
from urllib.parse import quote

from ...auth import build_auth
from ...embeddings import get_embedder
from ...errors import ConfigError
from ...plugins import HealthStatus, timed_health
from ...types import RetrievedDoc
from ...utils import expand_env
from ..base import RetrieverAdapter

_MODES = ("keyword", "vector", "hybrid", "semantic")
_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*$")
ENTRA_SCOPE = "https://search.azure.com/.default"


def odata_filter(f: dict | str | None) -> str | None:
    """dict -> OData filter with escaped literals; a string is used as-is."""
    if not f:
        return None
    if isinstance(f, str):
        return f

    def lit(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        if isinstance(v, (int, float)):
            return repr(v)
        return "'" + str(v).replace("'", "''") + "'"

    clauses = []
    for field, val in f.items():
        if not _FIELD.match(str(field)):
            raise ConfigError(f"azure_ai_search: invalid filter field name {field!r}")
        if isinstance(val, (list, tuple, set)):
            clauses.append("(" + " or ".join(f"{field} eq {lit(v)}" for v in val) + ")")
        else:
            clauses.append(f"{field} eq {lit(val)}")
    return " and ".join(clauses)


class AzureAISearchRetriever(RetrieverAdapter):
    def __init__(self, endpoint: str, index: str, mode: str = "keyword", api_key: str | None = None,
                 auth: dict | None = None, vector_fields: list[str] | str | None = None,
                 embedder: str | None = None, semantic_configuration: str | None = None,
                 text_field: str = "content", id_field: str = "id", source_field: str | None = None,
                 select: list[str] | None = None, search_fields: list[str] | None = None,
                 filter: dict | str | None = None, api_version: str = "2024-07-01",
                 timeout: float = 30, session=None, **_):
        if mode not in _MODES:
            raise ConfigError(f"azure_ai_search: mode must be one of {list(_MODES)}, got {mode!r}")
        if isinstance(vector_fields, str):
            vector_fields = [v.strip() for v in vector_fields.split(",") if v.strip()]
        self.vector_fields = list(vector_fields or [])
        if mode in ("vector", "hybrid") and not self.vector_fields:
            raise ConfigError(f"azure_ai_search: mode {mode} needs 'vector_fields' (the index's vector field names)")
        if mode == "semantic" and not semantic_configuration:
            raise ConfigError("azure_ai_search: mode semantic needs 'semantic_configuration' "
                              "(the name defined in the index's semantic settings)")
        self.endpoint = expand_env(endpoint).rstrip("/")
        self.index, self.mode, self.api_version = index, mode, api_version
        self.semantic_configuration = semantic_configuration
        self.text_field, self.id_field, self.source_field = text_field, id_field, source_field
        self.select, self.search_fields = select, search_fields
        self.filter = odata_filter(filter)
        self.timeout = timeout
        self.embed = get_embedder(embedder) if embedder else None

        self._owns = session is None
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        if auth:
            auth = dict(auth)
            if auth.get("type") == "entra_id":
                auth.setdefault("scope", ENTRA_SCOPE)
            self.session.auth = build_auth(auth)
        elif api_key:
            self.session.headers["api-key"] = expand_env(api_key)
        elif getattr(self.session, "auth", None) is None and "api-key" not in self.session.headers:
            raise ConfigError("azure_ai_search: set 'api_key' (e.g. ${AZURE_SEARCH_KEY}) or an auth "
                              "block, e.g. auth: { type: entra_id } (needs the 'Search Index Data "
                              "Reader' role on the search service)")

    @property
    def _docs_url(self) -> str:
        return f"{self.endpoint}/indexes/{quote(self.index, safe='')}/docs"

    def _post_search(self, body: dict) -> list[dict]:
        r = self.session.post(f"{self._docs_url}/search", params={"api-version": self.api_version},
                              json=body, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Azure AI Search query on '{self.index}' failed: HTTP {r.status_code} "
                               f"{r.text[:300]}")
        return r.json().get("value") or []

    def _vector_query(self, query: str, k: int) -> dict:
        fields = ",".join(self.vector_fields)
        if self.embed is not None:
            return {"kind": "vector", "vector": [float(x) for x in self.embed([query])[0]],
                    "fields": fields, "k": k}
        return {"kind": "text", "text": query, "fields": fields, "k": k}

    def build_body(self, query: str, k: int) -> dict:
        body: dict = {"top": k}
        if self.mode in ("keyword", "hybrid", "semantic"):
            body["search"] = query
        if self.mode in ("vector", "hybrid") or (self.mode == "semantic" and self.vector_fields):
            body["vectorQueries"] = [self._vector_query(query, k)]
        if self.mode == "semantic":
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = self.semantic_configuration
            body["captions"] = "extractive"
        if self.search_fields and "search" in body:
            body["searchFields"] = ",".join(self.search_fields)
        if self.select:
            body["select"] = ",".join(self.select)
        if self.filter:
            body["filter"] = self.filter
        return body

    def _to_doc(self, row: dict, with_score: bool = True) -> RetrievedDoc:
        row = dict(row)
        score = row.get("@search.rerankerScore")
        if score is None:
            score = row.get("@search.score")
        captions = row.get("@search.captions")
        meta = {key: v for key, v in row.items()
                if not key.startswith("@search.") and key not in self.vector_fields}
        text = meta.pop(self.text_field, "")
        doc_id = meta.get(self.id_field)
        if self.source_field and meta.get(self.source_field) is not None:
            meta.setdefault("source", meta[self.source_field])
        if captions:
            meta["captions"] = [c.get("text") for c in captions if isinstance(c, dict)]
        return RetrievedDoc(id=str(doc_id if doc_id is not None else ""),
                            text="" if text is None else str(text),
                            score=float(score) if (with_score and score is not None) else None,
                            metadata=meta)

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        return [self._to_doc(r) for r in self._post_search(self.build_body(query, k))]

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        # Reads a pool of up to 20n documents (max 1000 per request) and samples from it.
        body: dict = {"search": "*", "top": min(max(n * 20, n), 1000)}
        if self.select:
            body["select"] = ",".join(self.select)
        if self.filter:
            body["filter"] = self.filter
        pool = self._post_search(body)
        random.Random(seed).shuffle(pool)
        return [self._to_doc(r, with_score=False) for r in pool[:n]]

    def health_check(self) -> HealthStatus:
        def probe():
            r = self.session.get(f"{self._docs_url}/$count", params={"api-version": self.api_version},
                                 timeout=self.timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
            return f"{r.text.strip().lstrip(chr(0xFEFF))} doc(s) in {self.index}"
        return timed_health(probe)

    def close(self) -> None:
        if self._owns:
            self.session.close()
