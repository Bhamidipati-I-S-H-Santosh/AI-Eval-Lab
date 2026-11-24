"""Connect a Qdrant collection.

config:
  type: qdrant
  collection: company_docs
  url: http://localhost:6333          # or  path: ./qdrant_data  (local, no server)
  api_key: ${QDRANT_API_KEY}          # Qdrant Cloud only
  embedder: sentence-transformers:all-MiniLM-L6-v2   # must match how the index was built
  text_field: text                    # payload key holding the chunk text
  id_field: doc_id                    # payload key used as the doc id (default: point id)
  vector_name: null                   # set if the collection uses named vectors
  filter: { department: finance }     # optional fixed payload filter
"""
from __future__ import annotations

import random

from ..embeddings import get_embedder
from ..types import RetrievedDoc
from ..utils import expand_env
from .base import RetrieverAdapter


class QdrantRetrieverAdapter(RetrieverAdapter):
    def __init__(self, collection: str, url: str | None = None, path: str | None = None,
                 location: str | None = None, api_key: str | None = None,
                 embedder: str = "sentence-transformers:all-MiniLM-L6-v2",
                 text_field: str = "text", id_field: str | None = None,
                 vector_name: str | None = None, filter: dict | None = None,
                 client=None, **_):
        from qdrant_client import QdrantClient, models
        self.models = models
        if client is None:
            if path:
                client = QdrantClient(path=path)
            elif location:
                client = QdrantClient(location=location)
            else:
                client = QdrantClient(url=expand_env(url or "http://localhost:6333"),
                                      api_key=expand_env(api_key) or None)
        self.client = client
        self.collection = collection
        self.embed = get_embedder(embedder)
        self.text_field, self.id_field, self.vector_name = text_field, id_field, vector_name
        self.filter = self._build_filter(filter)

    def _build_filter(self, f: dict | None):
        if not f:
            return None
        m = self.models
        return m.Filter(must=[m.FieldCondition(key=k, match=m.MatchValue(value=v)) for k, v in f.items()])

    def _to_doc(self, point, score=None) -> RetrievedDoc:
        payload = dict(point.payload or {})
        doc_id = payload.get(self.id_field) if self.id_field else None
        text = payload.pop(self.text_field, "")
        return RetrievedDoc(id=str(doc_id if doc_id is not None else point.id), text=str(text),
                            score=score, metadata=payload)

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        vector = self.embed([query])[0]
        res = self.client.query_points(collection_name=self.collection, query=vector,
                                       using=self.vector_name, limit=k, query_filter=self.filter,
                                       with_payload=True)
        return [self._to_doc(p, p.score) for p in res.points]

    def close(self) -> None:
        self.client.close()

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        # Reads a pool of up to 20n points and samples from it. For very large
        # collections, sample by a payload field (e.g. random date ranges) instead.
        pool, offset = [], None
        while len(pool) < n * 20:
            points, offset = self.client.scroll(collection_name=self.collection, limit=256,
                                                offset=offset, scroll_filter=self.filter,
                                                with_payload=True)
            pool += points
            if offset is None:
                break
        random.Random(seed).shuffle(pool)
        return [self._to_doc(p) for p in pool[:n]]
