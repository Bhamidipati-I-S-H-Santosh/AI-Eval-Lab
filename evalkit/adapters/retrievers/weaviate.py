"""Connect a Weaviate collection (weaviate-client v4).

config:
  type: weaviate
  collection: CompanyDocs
  # self-hosted (connect_to_custom):
  url: http://localhost:8080
  grpc_port: 50051                    # grpc_host defaults to the url's host
  # or Weaviate Cloud (connect_to_weaviate_cloud):
  cluster_url: https://my-cluster.weaviate.cloud
  api_key: ${WEAVIATE_API_KEY}
  headers: { X-OpenAI-Api-Key: ${OPENAI_API_KEY} }   # for server-side vectorizers
  mode: near_vector                   # near_vector | near_text | hybrid | bm25
  alpha: 0.5                          # hybrid only: 1 = pure vector, 0 = pure keyword
  embedder: sentence-transformers:all-MiniLM-L6-v2   # near_vector (required) / hybrid (optional)
  text_field: text                    # property holding the chunk text
  id_field: null                      # property used as doc id (default: object uuid)
  source_field: source                # property copied to metadata["source"] if present
  target_vector: null                 # named vector, if the collection has several
  tenant: null                        # multi-tenant collections
  filter: { department: finance }     # optional property == value filters (ANDed)

near_text and hybrid without an embedder use the collection's own vectorizer.
score: 1 - distance for near_vector / near_text, Weaviate's score for hybrid / bm25.
"""
from __future__ import annotations

import random
from urllib.parse import urlparse

from ...embeddings import get_embedder
from ...errors import ConfigError, PluginDependencyError
from ...plugins import HealthStatus, timed_health
from ...types import RetrievedDoc
from ...utils import expand_env
from ..base import RetrieverAdapter

_MODES = ("near_vector", "near_text", "hybrid", "bm25")


def _query_classes():
    """(Filter, MetadataQuery) from weaviate.classes.query, or (None, None) when
    the SDK is not installed (only possible with an injected client)."""
    try:
        from weaviate.classes.query import Filter, MetadataQuery
        return Filter, MetadataQuery
    except ImportError:
        return None, None


class WeaviateRetriever(RetrieverAdapter):
    def __init__(self, collection: str, url: str | None = None, grpc_host: str | None = None,
                 grpc_port: int = 50051, cluster_url: str | None = None, api_key: str | None = None,
                 headers: dict | None = None, mode: str = "near_vector", alpha: float = 0.5,
                 embedder: str | None = None, text_field: str = "text", id_field: str | None = None,
                 source_field: str = "source", target_vector: str | None = None,
                 tenant: str | None = None, filter: dict | None = None, client=None, **_):
        if mode not in _MODES:
            raise ConfigError(f"weaviate: mode must be one of {list(_MODES)}, got {mode!r}")
        if mode == "near_vector" and not embedder:
            raise ConfigError("weaviate: mode near_vector needs 'embedder' (the model the collection "
                              "was built with), or use mode near_text to let Weaviate vectorize")
        self._owns = client is None
        if client is None:
            try:
                import weaviate
                from weaviate.classes.init import Auth
            except ImportError as e:
                raise PluginDependencyError("retriever:weaviate", "weaviate", "weaviate") from e
            auth = Auth.api_key(expand_env(api_key)) if api_key else None
            hdrs = expand_env(headers) if headers else None
            if cluster_url:
                client = weaviate.connect_to_weaviate_cloud(cluster_url=expand_env(cluster_url),
                                                            auth_credentials=auth, headers=hdrs)
            else:
                u = urlparse(expand_env(url or "http://localhost:8080"))
                secure = u.scheme == "https"
                client = weaviate.connect_to_custom(
                    http_host=u.hostname, http_port=u.port or (443 if secure else 80), http_secure=secure,
                    grpc_host=grpc_host or u.hostname, grpc_port=int(grpc_port), grpc_secure=secure,
                    headers=hdrs, auth_credentials=auth)
        self.client = client
        self.collection_name, self.mode, self.alpha = collection, mode, float(alpha)
        self.text_field, self.id_field, self.source_field = text_field, id_field, source_field
        self.target_vector, self.tenant = target_vector, tenant
        self.embed = get_embedder(embedder) if embedder else None
        self.Filter, self.MetadataQuery = _query_classes()
        self.filter = self._build_filter(filter)

    def _build_filter(self, f: dict | None):
        if not f:
            return None
        if self.Filter is None:
            raise PluginDependencyError("retriever:weaviate (filter)", "weaviate", "weaviate")
        parts = [self.Filter.by_property(k).equal(v) for k, v in f.items()]
        return parts[0] if len(parts) == 1 else self.Filter.all_of(parts)

    @property
    def collection(self):
        coll = self.client.collections.get(self.collection_name)
        return coll.with_tenant(self.tenant) if self.tenant else coll

    def _meta(self, **kw):
        return self.MetadataQuery(**kw) if self.MetadataQuery is not None else None

    def _to_doc(self, obj) -> RetrievedDoc:
        props = dict(getattr(obj, "properties", None) or {})
        text = props.pop(self.text_field, "")
        doc_id = props.get(self.id_field) if self.id_field else None
        if self.source_field and self.source_field != "source" and self.source_field in props:
            props.setdefault("source", props[self.source_field])
        md = getattr(obj, "metadata", None)
        dist, sc = getattr(md, "distance", None), getattr(md, "score", None)
        score = 1.0 - float(dist) if dist is not None else (float(sc) if sc is not None else None)
        return RetrievedDoc(id=str(doc_id if doc_id is not None else getattr(obj, "uuid", "")),
                            text="" if text is None else str(text), score=score, metadata=props)

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        q = self.collection.query
        common = {"limit": k, "filters": self.filter}
        if self.target_vector and self.mode != "bm25":
            common["target_vector"] = self.target_vector
        if self.mode == "near_vector":
            res = q.near_vector(near_vector=list(self.embed([query])[0]),
                                return_metadata=self._meta(distance=True), **common)
        elif self.mode == "near_text":
            res = q.near_text(query=query, return_metadata=self._meta(distance=True), **common)
        elif self.mode == "hybrid":
            vec = list(self.embed([query])[0]) if self.embed else None
            res = q.hybrid(query=query, vector=vec, alpha=self.alpha,
                           return_metadata=self._meta(score=True), **common)
        else:
            res = q.bm25(query=query, return_metadata=self._meta(score=True), **common)
        return [self._to_doc(o) for o in res.objects]

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        # Reads a pool of up to 20n objects and samples from it.
        res = self.collection.query.fetch_objects(limit=min(max(n * 20, n), 10_000), filters=self.filter)
        pool = list(res.objects)
        random.Random(seed).shuffle(pool)
        return [self._to_doc(o) for o in pool[:n]]

    def health_check(self) -> HealthStatus:
        def probe():
            agg = self.collection.aggregate.over_all(total_count=True)
            return f"{agg.total_count} object(s) in {self.collection_name}"
        return timed_health(probe)

    def close(self) -> None:
        if self._owns:
            self.client.close()
