"""Connect an Elasticsearch or OpenSearch index over plain REST (no SDK).

config:
  type: elasticsearch                 # or opensearch (same class; set flavor)
  flavor: elasticsearch               # elasticsearch | opensearch (decides the knn syntax)
  url: https://my-cluster.es.io:9243
  index: company-docs
  mode: bm25                          # bm25 | knn | hybrid (bm25 + knn fused with RRF)
  text_field: content                 # _source field holding the chunk text
  fields: [title^2, content]          # bm25 multi_match fields (default: [text_field])
  id_field: null                      # _source field used as doc id (default: _id)
  source_field: source                # _source field copied to metadata["source"] if present
  vector_field: embedding             # knn / hybrid
  embedder: sentence-transformers:all-MiniLM-L6-v2   # knn / hybrid; must match the index
  num_candidates: 100                 # Elasticsearch knn only
  hybrid_window: 50                   # hits fetched per leg before fusion
  rrf_k: 60
  filter: { department: finance }     # optional term filters (ANDed); list value -> terms
  auth:                               # evalkit auth block, e.g.
    type: api_key
    key: ${ES_API_KEY}
    header: Authorization
    prefix: "ApiKey "
  # AWS OpenSearch:  auth: { type: aws_sigv4, service: es, region: us-east-1 }   (serverless: service aoss)
  # basic auth:      username: elastic  /  password: ${ES_PASSWORD}
  verify_tls: true                    # or a CA bundle path
"""
from __future__ import annotations

from urllib.parse import quote

from ...auth import build_auth
from ...embeddings import get_embedder
from ...errors import ConfigError
from ...plugins import HealthStatus, timed_health
from ...types import RetrievedDoc
from ...utils import expand_env
from ..base import RetrieverAdapter

_MODES = ("bm25", "knn", "hybrid")
_FLAVORS = ("elasticsearch", "opensearch")


def rrf_fuse(ranked_lists: list[list[RetrievedDoc]], k: int, rrf_k: int = 60) -> list[RetrievedDoc]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (rrf_k + rank)."""
    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDoc] = {}
    for lst in ranked_lists:
        for rank, d in enumerate(lst, start=1):
            scores[d.id] = scores.get(d.id, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(d.id, d)
    order = sorted(scores, key=lambda i: -scores[i])[:k]
    return [RetrievedDoc(id=i, text=docs[i].text, score=scores[i], metadata=docs[i].metadata) for i in order]


class ElasticRetriever(RetrieverAdapter):
    def __init__(self, url: str, index: str, mode: str = "bm25", flavor: str = "elasticsearch",
                 text_field: str = "content", fields: list[str] | None = None,
                 id_field: str | None = None, source_field: str = "source",
                 vector_field: str | None = None, embedder: str | None = None,
                 num_candidates: int = 100, hybrid_window: int = 50, rrf_k: int = 60,
                 filter: dict | None = None, auth: dict | None = None,
                 username: str | None = None, password: str | None = None,
                 verify_tls: bool | str = True, timeout: float = 30, session=None, **_):
        if mode not in _MODES:
            raise ConfigError(f"elasticsearch: mode must be one of {list(_MODES)}, got {mode!r}")
        if flavor not in _FLAVORS:
            raise ConfigError(f"elasticsearch: flavor must be one of {list(_FLAVORS)}, got {flavor!r}")
        if mode in ("knn", "hybrid") and not (vector_field and embedder):
            raise ConfigError(f"elasticsearch: mode {mode} needs 'vector_field' and 'embedder' "
                              "(the model the index was built with)")
        self.url = expand_env(url).rstrip("/")
        self.index_path = quote(index, safe=",*")
        self.index, self.mode, self.flavor = index, mode, flavor
        self.text_field, self.fields = text_field, list(fields or [text_field])
        self.id_field, self.source_field, self.vector_field = id_field, source_field, vector_field
        self.num_candidates, self.hybrid_window, self.rrf_k = int(num_candidates), int(hybrid_window), int(rrf_k)
        self.filters = self._build_filters(filter)
        self.timeout = timeout
        self.embed = get_embedder(embedder) if embedder else None
        self._owns = session is None
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.session.verify = verify_tls
        provider = build_auth(auth)
        if provider is not None:
            self.session.auth = provider
        elif username:
            self.session.auth = (expand_env(username), expand_env(password or ""))

    @staticmethod
    def _build_filters(f: dict | None) -> list[dict]:
        out = []
        for field, val in (f or {}).items():
            if isinstance(val, (list, tuple, set)):
                out.append({"terms": {field: list(val)}})
            else:
                out.append({"term": {field: val}})
        return out

    # ---- HTTP ----
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        r = self.session.request(method, f"{self.url}/{self.index_path}/{path}", json=body,
                                 timeout=self.timeout, headers={"Content-Type": "application/json"})
        if r.status_code >= 400:
            raise RuntimeError(f"{self.flavor} {path} on index '{self.index}' failed: "
                               f"HTTP {r.status_code} {r.text[:300]}")
        return r.json()

    def _search(self, body: dict) -> list[RetrievedDoc]:
        if self.vector_field:
            body["_source"] = {"excludes": [self.vector_field]}
        res = self._request("POST", "_search", body)
        return [self._to_doc(h) for h in (res.get("hits") or {}).get("hits") or []]

    def _to_doc(self, hit: dict) -> RetrievedDoc:
        src = dict(hit.get("_source") or {})
        text = src.pop(self.text_field, "")
        if self.vector_field:
            src.pop(self.vector_field, None)
        doc_id = src.get(self.id_field) if self.id_field else None
        if self.source_field and self.source_field != "source" and self.source_field in src:
            src.setdefault("source", src[self.source_field])
        score = hit.get("_score")
        return RetrievedDoc(id=str(doc_id if doc_id is not None else hit.get("_id")),
                            text="" if text is None else str(text),
                            score=None if score is None else float(score), metadata=src)

    # ---- query bodies ----
    def bm25_body(self, query: str, size: int) -> dict:
        mm = {"multi_match": {"query": query, "fields": self.fields}}
        q = {"bool": {"must": [mm], "filter": self.filters}} if self.filters else mm
        return {"size": size, "query": q}

    def knn_body(self, query: str, size: int) -> dict:
        vec = [float(x) for x in self.embed([query])[0]]
        if self.flavor == "elasticsearch":
            knn = {"field": self.vector_field, "query_vector": vec, "k": size,
                   "num_candidates": max(self.num_candidates, size)}
            if self.filters:
                knn["filter"] = self.filters
            return {"size": size, "knn": knn}
        inner = {"vector": vec, "k": size}
        if self.filters:
            inner["filter"] = {"bool": {"filter": self.filters}}
        return {"size": size, "query": {"knn": {self.vector_field: inner}}}

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        if self.mode == "bm25":
            return self._search(self.bm25_body(query, k))
        if self.mode == "knn":
            return self._search(self.knn_body(query, k))
        window = max(k, self.hybrid_window)
        lexical = self._search(self.bm25_body(query, window))
        semantic = self._search(self.knn_body(query, window))
        return rrf_fuse([lexical, semantic], k, self.rrf_k)

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        # Server-side random sample, reproducible for a given seed.
        base = {"bool": {"filter": self.filters}} if self.filters else {"match_all": {}}
        body = {"size": n, "query": {"function_score": {
            "query": base, "random_score": {"seed": seed, "field": "_seq_no"}, "boost_mode": "replace"}}}
        docs = self._search(body)
        for d in docs:
            d.score = None
        return docs

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"{self._request('GET', '_count').get('count')} doc(s) in {self.index}")

    def close(self) -> None:
        if self._owns:
            self.session.close()


class OpenSearchRetriever(ElasticRetriever):
    """`type: opensearch`: the same REST retriever with the OpenSearch query syntax."""

    def __init__(self, *args, flavor: str = "opensearch", **kw):
        super().__init__(*args, flavor=flavor, **kw)
