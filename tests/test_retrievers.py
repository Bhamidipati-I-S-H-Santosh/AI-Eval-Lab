"""Retriever plugins, tested with fake clients that mimic each vendor's
response shapes. No network, no vendor SDKs required."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from evalkit.errors import ConfigError, PluginDependencyError
from evalkit.plugins import REGISTRY
from evalkit.types import RetrievedDoc

EMB = "hash:16"


# ---------------------------------------------------------------- fakes
class FakeResponse:
    def __init__(self, body=None, status=200, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    """Records requests; applies session.auth like requests does; replies via a handler."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []
        self.headers: dict = {}
        self.auth = None
        self.verify = True
        self.closed = False

    def request(self, method, url, json=None, params=None, timeout=None, headers=None, **_):
        hdrs = {**self.headers, **(headers or {})}
        if callable(self.auth):
            req = SimpleNamespace(method=method, url=url, headers=hdrs, body=json)
            self.auth(req)
        call = {"method": method, "url": url, "json": json, "params": params, "headers": hdrs}
        self.calls.append(call)
        return self.handler(call)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def close(self):
        self.closed = True


# ================================================================ pgvector
class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, list(params or [])))
        if self.conn.fail:
            raise RuntimeError("connection refused")

    def fetchall(self):
        return self.conn.rows


class FakeConn:
    def __init__(self, rows=None, fail=False):
        self.rows, self.fail, self.executed, self.closed = rows or [], fail, [], False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


def _pg(conn, **kw):
    from evalkit.adapters.retrievers.pgvector import PgVectorRetriever
    opts = dict(table="public.chunks", text_column="content", metadata_columns=["title"],
                source_column="url", embedder=EMB, connection=conn)
    opts.update(kw)
    return PgVectorRetriever(**opts)


def test_pgvector_search_maps_rows_and_parameterizes():
    conn = FakeConn(rows=[(1, "alpha text", "Alpha", "a.md", 0.1), (2, "beta text", "Beta", None, 0.4)])
    r = _pg(conn, filter={"tenant": "acme", "lang": ["en", "de"]})
    docs = r.search("what is alpha?", k=2)
    assert [d.id for d in docs] == ["1", "2"]
    assert docs[0].text == "alpha text"
    assert docs[0].score == pytest.approx(0.9)       # cosine: 1 - distance
    assert docs[0].metadata == {"title": "Alpha", "source": "a.md"}
    assert "source" not in docs[1].metadata
    sql, params = conn.executed[0]
    assert '"embedding" <=> %s::vector' in sql
    assert 'FROM "public"."chunks" WHERE "tenant" = %s AND "lang" = ANY(%s)' in sql
    assert "ORDER BY _distance LIMIT %s" in sql
    assert params[0].startswith("[") and params[1:] == ["acme", ["en", "de"], 2]
    assert "acme" not in sql and "alpha" not in sql


@pytest.mark.parametrize("distance,op,score", [("l2", "<->", -0.5), ("inner_product", "<#>", -0.5)])
def test_pgvector_distance_operators(distance, op, score):
    conn = FakeConn(rows=[(1, "t", "T", "s", 0.5)])
    docs = _pg(conn, distance=distance).search("q", 1)
    assert f'"embedding" {op} %s::vector' in conn.executed[0][0]
    assert docs[0].score == pytest.approx(score)


@pytest.mark.parametrize("bad", [
    {"table": "chunks; DROP TABLE users"},
    {"table": 'chunks"--'},
    {"table": "a.b.c"},
    {"text_column": "content) OR 1=1 --"},
    {"embedding_column": "emb bedding"},
    {"metadata_columns": ["title", "x;y"]},
    {"source_column": "1url"},
    {"filter": {"tenant = 'x' OR 1=1 --": "acme"}},
    {"distance": "manhattan"},
])
def test_pgvector_rejects_unsafe_identifiers(bad):
    with pytest.raises(ConfigError):
        _pg(FakeConn(), **bad)


def test_pgvector_sample_and_health_and_close():
    conn = FakeConn(rows=[(3, "gamma", "G", "g.md")])
    r = _pg(conn, filter={"tenant": "acme"})
    docs = r.sample(1, seed=7)
    assert docs[0].id == "3" and docs[0].score is None and docs[0].metadata["source"] == "g.md"
    assert conn.executed[0][0] == "SELECT setseed(%s)" and -1 <= conn.executed[0][1][0] <= 1
    assert "ORDER BY random() LIMIT %s" in conn.executed[1][0] and conn.executed[1][1] == ["acme", 1]
    assert r.health_check().ok
    r.close()
    assert not conn.closed                     # injected connection is the caller's
    assert not _pg(FakeConn(fail=True)).health_check().ok


def test_pgvector_without_psycopg_raises_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(PluginDependencyError):
        REGISTRY.build("retriever", {"type": "pgvector", "dsn": "postgresql://x", "table": "t",
                                     "embedder": EMB})


# ================================================================ pinecone
class FakePineconeIndex:
    def __init__(self, fail=False):
        self.queries, self.fail = [], fail
        self.vectors = {f"v{i}": SimpleNamespace(id=f"v{i}", metadata={"text": f"doc {i}", "source": f"f{i}.pdf"})
                        for i in range(30)}

    def query(self, **kw):
        self.queries.append(kw)
        return SimpleNamespace(matches=[
            SimpleNamespace(id="v2", score=0.91, metadata={"text": "two", "source": "two.pdf", "page": 3}),
            SimpleNamespace(id="v1", score=0.55, metadata={"text": "one", "doc_id": "D1"}),
        ])

    def list(self, namespace=None):
        ids = sorted(self.vectors)
        for i in range(0, len(ids), 10):
            yield ids[i:i + 10]

    def fetch(self, ids, namespace=None):
        return SimpleNamespace(vectors={i: self.vectors[i] for i in ids})

    def describe_index_stats(self):
        if self.fail:
            raise RuntimeError("401 unauthorized")
        return {"total_vector_count": 30}


def test_pinecone_via_registry_search_mapping_and_filter():
    idx = FakePineconeIndex()
    r = REGISTRY.build("retriever", {"type": "pinecone", "index_name": "docs", "namespace": "prod",
                                     "filter": {"dept": {"$eq": "hr"}}, "embedder": EMB, "index": idx})
    docs = r.search("q", k=2)
    assert [d.id for d in docs] == ["v2", "v1"]
    assert docs[0].text == "two" and docs[0].score == pytest.approx(0.91)
    assert docs[0].metadata == {"source": "two.pdf", "page": 3}
    q = idx.queries[0]
    assert q["top_k"] == 2 and q["include_metadata"] and q["namespace"] == "prod"
    assert q["filter"] == {"dept": {"$eq": "hr"}} and len(q["vector"]) == 16


def test_pinecone_id_field_dict_response_sample_health():
    from evalkit.adapters.retrievers.pinecone import PineconeRetriever

    class DictIndex(FakePineconeIndex):
        def query(self, **kw):
            return {"matches": [{"id": "v1", "score": 0.5, "metadata": {"text": "one", "doc_id": "D1"}}]}

    r = PineconeRetriever(index=DictIndex(), id_field="doc_id", embedder=EMB)
    assert r.search("q")[0].id == "D1"
    s1, s2 = r.sample(5, seed=1), r.sample(5, seed=1)
    assert [d.id for d in s1] == [d.id for d in s2] and len(s1) == 5
    assert all(d.text.startswith("doc ") and d.metadata["source"].endswith(".pdf") for d in s1)
    assert r.health_check().ok and "30" in r.health_check().detail
    assert not PineconeRetriever(index=FakePineconeIndex(fail=True), embedder=EMB).health_check().ok


def test_pinecone_without_sdk_raises_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pinecone", None)
    with pytest.raises(PluginDependencyError):
        REGISTRY.build("retriever", {"type": "pinecone", "index_name": "docs", "api_key": "k",
                                     "embedder": EMB})


# ================================================================ weaviate
class _FakeFilter:
    def __init__(self, desc):
        self.desc = desc

    @classmethod
    def by_property(cls, name):
        return SimpleNamespace(equal=lambda v: cls(("eq", name, v)))

    @classmethod
    def all_of(cls, parts):
        return cls(("and", [p.desc for p in parts]))


class _FakeMetadataQuery:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture
def fake_weaviate_classes(monkeypatch):
    for name in ("weaviate", "weaviate.classes"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())
    monkeypatch.setitem(sys.modules, "weaviate.classes.query",
                        SimpleNamespace(Filter=_FakeFilter, MetadataQuery=_FakeMetadataQuery))


def _wobj(uuid, props, distance=None, score=None):
    return SimpleNamespace(uuid=uuid, properties=props, metadata=SimpleNamespace(distance=distance, score=score))


class FakeWeaviateClient:
    def __init__(self, fail=False):
        self.calls, self.closed, self.fail, self.tenant = [], False, fail, None
        outer = self

        class Q:
            def _rec(self, name, kw):
                outer.calls.append((name, kw))
                if name in ("near_vector", "near_text"):
                    objs = [_wobj("u1", {"text": "first", "source": "a.md"}, distance=0.2),
                            _wobj("u2", {"text": "second"}, distance=0.5)]
                elif name == "fetch_objects":
                    objs = [_wobj(f"u{i}", {"text": f"t{i}"}) for i in range(10)]
                else:
                    objs = [_wobj("u9", {"text": "kw", "url": "x.html"}, score=1.7)]
                return SimpleNamespace(objects=objs)

            def near_vector(self, **kw): return self._rec("near_vector", kw)
            def near_text(self, **kw): return self._rec("near_text", kw)
            def hybrid(self, **kw): return self._rec("hybrid", kw)
            def bm25(self, **kw): return self._rec("bm25", kw)
            def fetch_objects(self, **kw): return self._rec("fetch_objects", kw)

        class Agg:
            def over_all(self, total_count=False):
                if outer.fail:
                    raise RuntimeError("unreachable")
                return SimpleNamespace(total_count=10)

        class Coll:
            query, aggregate = Q(), Agg()

            def with_tenant(self, t):
                outer.tenant = t
                return self

        self.collections = SimpleNamespace(get=lambda name: (outer.calls.append(("get", name)), Coll())[1])

    def close(self):
        self.closed = True


def test_weaviate_near_vector_mapping_filter_and_tenant(fake_weaviate_classes):
    from evalkit.adapters.retrievers.weaviate import WeaviateRetriever
    c = FakeWeaviateClient()
    r = WeaviateRetriever(collection="Docs", embedder=EMB, filter={"dept": "hr", "lang": "en"},
                          tenant="acme", client=c)
    docs = r.search("q", k=2)
    assert [d.id for d in docs] == ["u1", "u2"]
    assert docs[0].text == "first" and docs[0].score == pytest.approx(0.8)
    assert docs[0].metadata == {"source": "a.md"}
    name, kw = [x for x in c.calls if x[0] == "near_vector"][0]
    assert kw["limit"] == 2 and len(kw["near_vector"]) == 16
    assert kw["filters"].desc == ("and", [("eq", "dept", "hr"), ("eq", "lang", "en")])
    assert kw["return_metadata"].kw == {"distance": True}
    assert c.tenant == "acme" and ("get", "Docs") in c.calls


def test_weaviate_modes_and_sample_health(fake_weaviate_classes):
    from evalkit.adapters.retrievers.weaviate import WeaviateRetriever
    c = FakeWeaviateClient()
    r = WeaviateRetriever(collection="Docs", mode="hybrid", alpha=0.3, source_field="url", client=c)
    d = r.search("q", 3)[0]
    assert d.id == "u9" and d.score == pytest.approx(1.7) and d.metadata["source"] == "x.html"
    kw = [x for x in c.calls if x[0] == "hybrid"][0][1]
    assert kw["query"] == "q" and kw["alpha"] == 0.3 and kw["vector"] is None and kw["filters"] is None
    WeaviateRetriever(collection="Docs", mode="near_text", client=c).search("hello", 1)
    assert [x for x in c.calls if x[0] == "near_text"][0][1]["query"] == "hello"
    s = r.sample(4, seed=3)
    assert len(s) == 4 and [x.id for x in s] == [x.id for x in r.sample(4, seed=3)]
    assert r.health_check().ok
    assert not WeaviateRetriever(collection="Docs", mode="bm25", client=FakeWeaviateClient(fail=True)).health_check().ok
    r.close()
    assert not c.closed


def test_weaviate_config_errors_and_lazy_import(monkeypatch):
    from evalkit.adapters.retrievers.weaviate import WeaviateRetriever
    with pytest.raises(ConfigError):
        WeaviateRetriever(collection="Docs", mode="near_vector", client=FakeWeaviateClient())
    with pytest.raises(ConfigError):
        WeaviateRetriever(collection="Docs", mode="graph", client=FakeWeaviateClient())
    monkeypatch.setitem(sys.modules, "weaviate", None)
    with pytest.raises(PluginDependencyError):
        REGISTRY.build("retriever", {"type": "weaviate", "collection": "Docs", "mode": "near_text"})


# ================================================================ elastic / opensearch
def _es_hits(*rows):
    return {"hits": {"hits": [{"_id": i, "_score": s, "_source": src} for i, s, src in rows]}}


def _es_handler(call):
    if call["url"].endswith("/_count"):
        return FakeResponse({"count": 42})
    body = call["json"]
    if "knn" in body or "knn" in (body.get("query") or {}):
        return FakeResponse(_es_hits(("c", 0.9, {"content": "C"}), ("a", 0.8, {"content": "A"}),
                                     ("d", 0.7, {"content": "D"})))
    if "function_score" in (body.get("query") or {}):
        return FakeResponse(_es_hits(("s1", 1.0, {"content": "S1", "doc_id": "DOC-1"})))
    return FakeResponse(_es_hits(("a", 12.0, {"content": "A", "source": "a.md", "embedding": [1, 2]}),
                                 ("b", 9.0, {"content": "B"}), ("c", 3.0, {"content": "C"})))


def test_elastic_bm25_with_api_key_auth_and_filters():
    sess = FakeSession(_es_handler)
    r = REGISTRY.build("retriever", {
        "type": "elasticsearch", "url": "https://es.example:9243/", "index": "docs",
        "filter": {"dept": "hr", "lang": ["en", "de"]}, "vector_field": "embedding", "verify_tls": False,
        "auth": {"type": "api_key", "key": "SECRET", "header": "Authorization", "prefix": "ApiKey "},
        "session": sess})
    docs = r.search("vacation policy", k=3)
    assert [d.id for d in docs] == ["a", "b", "c"]
    assert docs[0].text == "A" and docs[0].score == 12.0 and docs[0].metadata == {"source": "a.md"}
    call = sess.calls[0]
    assert call["url"] == "https://es.example:9243/docs/_search"
    assert call["headers"]["Authorization"] == "ApiKey SECRET"
    assert sess.verify is False
    q = call["json"]["query"]["bool"]
    assert q["must"][0]["multi_match"] == {"query": "vacation policy", "fields": ["content"]}
    assert q["filter"] == [{"term": {"dept": "hr"}}, {"terms": {"lang": ["en", "de"]}}]
    assert call["json"]["size"] == 3 and call["json"]["_source"] == {"excludes": ["embedding"]}


def test_elastic_knn_flavors():
    sess = FakeSession(_es_handler)
    r = REGISTRY.build("retriever", {"type": "elasticsearch", "url": "http://es", "index": "docs", "mode": "knn",
                                     "vector_field": "embedding", "embedder": EMB, "filter": {"dept": "hr"},
                                     "session": sess})
    assert [d.id for d in r.search("q", 2)] == ["c", "a", "d"]  # fake ignores size
    knn = sess.calls[0]["json"]["knn"]
    assert knn["field"] == "embedding" and knn["k"] == 2 and knn["num_candidates"] == 100
    assert len(knn["query_vector"]) == 16 and knn["filter"] == [{"term": {"dept": "hr"}}]

    sess2 = FakeSession(_es_handler)
    r2 = REGISTRY.build("retriever", {"type": "opensearch", "flavor": "opensearch", "url": "http://os",
                                      "index": "docs", "mode": "knn", "vector_field": "emb",
                                      "embedder": EMB, "filter": {"dept": "hr"}, "session": sess2})
    r2.search("q", 4)
    body = sess2.calls[0]["json"]
    assert "knn" not in body
    inner = body["query"]["knn"]["emb"]
    assert inner["k"] == 4 and len(inner["vector"]) == 16
    assert inner["filter"] == {"bool": {"filter": [{"term": {"dept": "hr"}}]}}


def test_elastic_hybrid_rrf_fusion_order():
    from evalkit.adapters.retrievers.elastic import rrf_fuse
    sess = FakeSession(_es_handler)
    r = REGISTRY.build("retriever", {"type": "elasticsearch", "url": "http://es", "index": "docs",
                                     "mode": "hybrid", "vector_field": "embedding", "embedder": EMB,
                                     "hybrid_window": 10, "session": sess})
    docs = r.search("q", k=3)
    # bm25: a,b,c   knn: c,a,d  -> a: 1/61+1/62, c: 1/63+1/61, b: 1/62, d: 1/63
    assert [d.id for d in docs] == ["a", "c", "b"]
    assert docs[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert len(sess.calls) == 2 and all(c["json"]["size"] == 10 for c in sess.calls)
    fused = rrf_fuse([[RetrievedDoc("x", "")], [RetrievedDoc("y", ""), RetrievedDoc("x", "")]], k=5)
    assert [d.id for d in fused] == ["x", "y"]


def test_elastic_sample_health_errors():
    sess = FakeSession(_es_handler)
    r = REGISTRY.build("retriever", {"type": "elasticsearch", "url": "http://es", "index": "docs",
                                     "id_field": "doc_id", "session": sess})
    s = r.sample(1, seed=5)
    assert s[0].id == "DOC-1" and s[0].text == "S1" and s[0].score is None
    fs = sess.calls[0]["json"]["query"]["function_score"]
    assert fs["random_score"] == {"seed": 5, "field": "_seq_no"} and fs["query"] == {"match_all": {}}
    h = r.health_check()
    assert h.ok and "42" in h.detail
    bad = REGISTRY.build("retriever", {"type": "elasticsearch", "url": "http://es", "index": "docs",
                                       "session": FakeSession(lambda c: FakeResponse({"error": "x"}, 401))})
    assert not bad.health_check().ok
    with pytest.raises(RuntimeError, match="HTTP 401"):
        bad.search("q")
    with pytest.raises(ConfigError):
        REGISTRY.build("retriever", {"type": "elasticsearch", "url": "http://es", "index": "d", "mode": "knn",
                                     "session": FakeSession(_es_handler)})


def test_elastic_aws_sigv4_auth_is_attached():
    from evalkit.auth import AwsSigV4Auth
    sess = FakeSession(_es_handler)
    fake_boto = SimpleNamespace(get_credentials=lambda: None)
    r = REGISTRY.build("retriever", {"type": "opensearch", "flavor": "opensearch", "url": "https://os.aws",
                                     "index": "docs", "session": sess,
                                     "auth": {"type": "aws_sigv4", "service": "es", "region": "us-east-1",
                                              "session": fake_boto}})
    assert isinstance(sess.auth, AwsSigV4Auth) and sess.auth.service == "es"


# ================================================================ azure ai search
def _az_handler(call):
    if call["url"].endswith("/$count"):
        return FakeResponse(None, text="﻿1234")
    return FakeResponse({"value": [
        {"@search.score": 0.03, "@search.rerankerScore": 2.9, "id": "doc-1", "content": "Leave policy",
         "metadata_storage_name": "leave.pdf", "contentVector": [0.1],
         "@search.captions": [{"text": "Leave is 20 days", "highlights": ""}]},
        {"@search.score": 0.02, "id": "doc-2", "content": "Pay policy", "metadata_storage_name": "pay.pdf"},
    ]})


def _az(mode="keyword", **kw):
    sess = FakeSession(_az_handler)
    cfg = {"type": "azure_ai_search", "endpoint": "https://s.search.windows.net/", "index": "docs",
           "mode": mode, "api_key": "AZKEY", "source_field": "metadata_storage_name", "session": sess}
    cfg.update(kw)
    return REGISTRY.build("retriever", cfg), sess


def test_azure_keyword_request_and_mapping():
    r, sess = _az(filter={"category": "O'Brien", "year": 2024}, select=["id", "content"])
    docs = r.search("leave", 2)
    call = sess.calls[0]
    assert call["url"] == "https://s.search.windows.net/indexes/docs/docs/search"
    assert call["params"] == {"api-version": "2024-07-01"}
    assert call["headers"]["api-key"] == "AZKEY"
    assert call["json"] == {"top": 2, "search": "leave", "select": "id,content",
                            "filter": "category eq 'O''Brien' and year eq 2024"}
    assert [d.id for d in docs] == ["doc-1", "doc-2"]
    assert docs[1].text == "Pay policy" and docs[1].score == pytest.approx(0.02)
    assert docs[1].metadata["source"] == "pay.pdf"


def test_azure_vector_hybrid_semantic_bodies():
    r, sess = _az("vector", vector_fields=["contentVector"], embedder=EMB)
    r.search("q", 3)
    vq = sess.calls[0]["json"]["vectorQueries"][0]
    assert "search" not in sess.calls[0]["json"]
    assert vq["kind"] == "vector" and vq["fields"] == "contentVector" and vq["k"] == 3 and len(vq["vector"]) == 16

    r, sess = _az("hybrid", vector_fields="contentVector,titleVector")
    docs = r.search("q", 3)
    body = sess.calls[0]["json"]
    assert body["search"] == "q"
    assert body["vectorQueries"] == [{"kind": "text", "text": "q", "fields": "contentVector,titleVector", "k": 3}]
    assert "contentVector" not in docs[0].metadata

    r, sess = _az("semantic", semantic_configuration="sem1")
    docs = r.search("q", 2)
    body = sess.calls[0]["json"]
    assert body["queryType"] == "semantic" and body["semanticConfiguration"] == "sem1"
    assert "vectorQueries" not in body
    assert docs[0].score == pytest.approx(2.9) and docs[0].metadata["captions"] == ["Leave is 20 days"]


def test_azure_entra_auth_sample_health_and_errors():
    cred = SimpleNamespace(get_token=lambda scope: SimpleNamespace(token=f"TOK[{scope}]", expires_on=9e9))
    r, sess = _az(api_key=None, auth={"type": "entra_id", "credential": cred})
    r.search("q")
    assert sess.calls[0]["headers"]["Authorization"] == "Bearer TOK[https://search.azure.com/.default]"
    s = r.sample(1, seed=2)
    assert len(s) == 1 and s[0].score is None
    assert sess.calls[-1]["json"]["search"] == "*" and sess.calls[-1]["json"]["top"] == 20
    h = r.health_check()
    assert h.ok and "1234" in h.detail and sess.calls[-1]["url"].endswith("/docs/$count")
    with pytest.raises(ConfigError):
        _az(api_key=None)
    with pytest.raises(ConfigError):
        _az("vector")
    with pytest.raises(ConfigError):
        _az("semantic")
    with pytest.raises(ConfigError):
        _az(filter={"x eq 1 or y": 1})
    bad, _ = _az(session=FakeSession(lambda c: FakeResponse({"error": "no"}, 403)))
    assert not bad.health_check().ok


# ================================================================ vertex ai search
def _vx_handler(call):
    if call["method"] == "GET":
        return FakeResponse({"documents": [
            {"id": f"d{i}", "structData": {"body": f"text {i}", "src": f"s{i}"}} for i in range(5)]})
    return FakeResponse({"results": [
        {"id": "doc-a", "document": {
            "name": "projects/p/.../documents/doc-a", "id": "doc-a",
            "derivedStructData": {"title": "Handbook", "link": "gs://bucket/handbook.pdf",
                                  "snippets": [{"snippet": "Employees get <b>20</b> days", "snippet_status": "SUCCESS"}],
                                  "extractive_answers": [{"content": "20 days of leave", "pageNumber": "4"}]}}},
        {"id": "doc-b", "document": {"id": "doc-b", "structData": {"body": "structured text", "src": "crm"}}},
    ]})


class FakeAuth:
    def __call__(self, req):
        req.headers["Authorization"] = "Bearer GCPTOKEN"
        return req


def _vx(**kw):
    sess = FakeSession(_vx_handler)
    sess.auth = FakeAuth()
    cfg = {"type": "vertex_search", "project": "my-proj", "engine_id": "app1", "session": sess}
    cfg.update(kw)
    return REGISTRY.build("retriever", cfg), sess


def test_vertex_search_request_and_mapping():
    r, sess = _vx(extractive_answers=1, filter='lang: ANY("en")', quota_project="billing-proj")
    docs = r.search("leave days", 2)
    call = sess.calls[0]
    assert call["url"] == ("https://discoveryengine.googleapis.com/v1/projects/my-proj/locations/global/"
                           "collections/default_collection/engines/app1/servingConfigs/default_search:search")
    assert call["headers"]["Authorization"] == "Bearer GCPTOKEN"
    assert call["headers"]["x-goog-user-project"] == "billing-proj"
    assert call["json"] == {"query": "leave days", "pageSize": 2, "filter": 'lang: ANY("en")',
                            "contentSearchSpec": {"snippetSpec": {"returnSnippet": True},
                                                  "extractiveContentSpec": {"maxExtractiveAnswerCount": 1}}}
    assert [d.id for d in docs] == ["doc-a", "doc-b"]
    assert docs[0].text == "20 days of leave"
    assert docs[0].metadata["source"] == "gs://bucket/handbook.pdf" and docs[0].metadata["rank"] == 1
    assert docs[0].metadata["title"] == "Handbook"
    assert docs[1].metadata["source"] == "doc-b"


def test_vertex_snippet_fallback_text_field_and_data_store():
    r, sess = _vx(engine_id=None, data_store_id="ds1", location="eu", text_field="body", source_field="src")
    assert sess.calls == [] and r.search_url.startswith(
        "https://eu-discoveryengine.googleapis.com/v1/projects/my-proj/locations/eu/collections/"
        "default_collection/dataStores/ds1/servingConfigs/default_search:search")
    docs = r.search("q")
    assert docs[0].text == "20 days of leave"               # extractive answer preferred
    snippet_only = {"snippets": [{"snippet": "Employees get <b>20</b> days", "snippet_status": "SUCCESS"},
                                 {"snippet": "No snippet", "snippet_status": "NO_SNIPPET_AVAILABLE"}]}
    assert r._text({}, snippet_only) == "Employees get 20 days"   # HTML stripped, failures skipped
    assert docs[1].text == "structured text" and docs[1].metadata["source"] == "crm"
    assert "body" not in docs[1].metadata
    s = r.sample(3, seed=4)
    assert len(s) == 3 and all(d.text.startswith("text ") for d in s)
    assert [d.id for d in s] == [d.id for d in r.sample(3, seed=4)]
    assert sess.calls[1]["url"].endswith("/dataStores/ds1/branches/default_branch/documents")
    assert r.health_check().ok


def test_vertex_config_errors_and_sample_needs_data_store():
    r, _ = _vx()
    with pytest.raises(NotImplementedError):
        r.sample(2)
    with pytest.raises(ConfigError):
        _vx(engine_id=None)
    with pytest.raises(ConfigError):
        _vx(data_store_id="ds")
    bad, _ = _vx(session=FakeSession(lambda c: FakeResponse({"error": {"message": "denied"}}, 403)))
    assert not bad.health_check().ok


def test_vertex_defaults_to_gcp_service_account_auth():
    from evalkit.auth import GcpServiceAccountAuth
    sess = FakeSession(_vx_handler)
    REGISTRY.build("retriever", {"type": "vertex_search", "project": "p", "engine_id": "e", "session": sess})
    assert isinstance(sess.auth, GcpServiceAccountAuth)


# ================================================================ registry
def test_all_retrievers_registered_and_importable_without_sdks():
    names = {s.name for s in REGISTRY.list("retriever")}
    for n in ("pgvector", "pinecone", "weaviate", "elasticsearch", "opensearch", "azure_ai_search", "vertex_search"):
        assert n in names
        cls = REGISTRY.get("retriever", n)
        assert hasattr(cls, "search") and hasattr(cls, "sample")
