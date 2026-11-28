"""Connect a Pinecone index (pinecone SDK v5+).

config:
  type: pinecone
  api_key: ${PINECONE_API_KEY}
  index_name: company-docs            # or  host: company-docs-abc123.svc.us-east-1.pinecone.io
  namespace: ""                       # optional
  text_field: text                    # metadata key holding the chunk text
  id_field: null                      # metadata key used as doc id (default: vector id)
  source_field: source                # metadata key copied to metadata["source"] if present
  filter: { department: finance }     # optional Pinecone metadata filter (passed as-is)
  embedder: sentence-transformers:all-MiniLM-L6-v2   # must match how the index was built

sample() uses index.list() + fetch(), which Pinecone supports on serverless
indexes only. It ignores `filter`.
"""
from __future__ import annotations

import os
import random

from ...embeddings import get_embedder
from ...errors import ConfigError, PluginDependencyError
from ...plugins import HealthStatus, timed_health
from ...types import RetrievedDoc
from ...utils import expand_env
from ..base import RetrieverAdapter


def _get(obj, key, default=None):
    """Read a field from an SDK object or a plain dict response."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class PineconeRetriever(RetrieverAdapter):
    def __init__(self, index_name: str | None = None, host: str | None = None,
                 api_key: str | None = None, namespace: str | None = None,
                 text_field: str = "text", id_field: str | None = None,
                 source_field: str = "source", filter: dict | None = None,
                 embedder: str = "sentence-transformers:all-MiniLM-L6-v2",
                 index=None, **_):
        if index is None:
            if not (index_name or host):
                raise ConfigError("pinecone: set 'index_name' or 'host' (or inject 'index')")
            key = expand_env(api_key) if api_key else os.getenv("PINECONE_API_KEY")
            if not key:
                raise ConfigError("pinecone: set 'api_key' (e.g. ${PINECONE_API_KEY})")
            try:
                from pinecone import Pinecone
            except ImportError as e:
                raise PluginDependencyError("retriever:pinecone", "pinecone", "pinecone") from e
            pc = Pinecone(api_key=key)
            index = pc.Index(host=host) if host else pc.Index(index_name)
        self.index = index
        self.namespace = namespace or None
        self.text_field, self.id_field, self.source_field = text_field, id_field, source_field
        self.filter = filter or None
        self.embed = get_embedder(embedder)

    def _to_doc(self, vec_id, metadata, score=None) -> RetrievedDoc:
        meta = dict(metadata or {})
        text = meta.pop(self.text_field, "")
        doc_id = meta.get(self.id_field) if self.id_field else None
        if self.source_field and self.source_field != "source" and self.source_field in meta:
            meta.setdefault("source", meta[self.source_field])
        return RetrievedDoc(id=str(doc_id if doc_id is not None else vec_id),
                            text="" if text is None else str(text),
                            score=None if score is None else float(score), metadata=meta)

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        kwargs = {"vector": list(self.embed([query])[0]), "top_k": k, "include_metadata": True}
        if self.namespace:
            kwargs["namespace"] = self.namespace
        if self.filter:
            kwargs["filter"] = self.filter
        res = self.index.query(**kwargs)
        return [self._to_doc(_get(m, "id"), _get(m, "metadata"), _get(m, "score"))
                for m in (_get(res, "matches") or [])]

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        # index.list() yields pages (lists) of ids; serverless indexes only.
        kw = {"namespace": self.namespace} if self.namespace else {}
        ids: list[str] = []
        try:
            for page in self.index.list(**kw):
                ids += list(page) if not isinstance(page, str) else [page]
                if len(ids) >= n * 20:
                    break
        except Exception as e:
            raise NotImplementedError(f"PineconeRetriever cannot sample: index.list() failed ({e}). "
                                      "Listing ids is supported on serverless indexes only") from e
        random.Random(seed).shuffle(ids)
        chosen = ids[:n]
        if not chosen:
            return []
        res = self.index.fetch(ids=chosen, **kw)
        vectors = _get(res, "vectors") or {}
        return [self._to_doc(i, _get(vectors[i], "metadata")) for i in chosen if i in vectors]

    def health_check(self) -> HealthStatus:
        def probe():
            stats = self.index.describe_index_stats()
            total = _get(stats, "total_vector_count")
            return f"{total} vector(s) in index"
        return timed_health(probe)
