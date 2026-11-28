"""Connect a Vertex AI Search (Discovery Engine) app or data store over REST.

config:
  type: vertex_search
  project: my-gcp-project
  location: global                    # global | us | eu  (us/eu use the regional endpoint)
  engine_id: company-docs-app         # a search app ...
  # data_store_id: company-docs-ds    # ... or a data store directly (also enables sample())
  serving_config: default_search
  text_field: null                    # structData field holding the text (structured data);
                                      # otherwise extractive segments/answers or snippets are used
  id_field: null                      # structData field used as doc id (default: document id)
  source_field: null                  # structData field copied to metadata["source"]
                                      # (default: derivedStructData.link, e.g. gs://bucket/file.pdf)
  snippets: true                      # contentSearchSpec.snippetSpec.returnSnippet
  extractive_answers: 0               # maxExtractiveAnswerCount (Enterprise edition)
  extractive_segments: 0              # maxExtractiveSegmentCount (Enterprise edition)
  filter: 'category: ANY("finance")'  # optional Discovery Engine filter expression
  quota_project: null                 # x-goog-user-project header (needed with user ADC creds)
  auth: { type: gcp_service_account } # default; any evalkit auth block works

Results carry no relevance score by default, so score is None and the order is
Vertex's ranking (metadata["rank"] is 1-based).
"""
from __future__ import annotations

import json
import random
import re
from urllib.parse import quote

from ...auth import GcpServiceAccountAuth, build_auth
from ...errors import ConfigError
from ...plugins import HealthStatus, timed_health
from ...types import RetrievedDoc
from ..base import RetrieverAdapter

_TAG = re.compile(r"<[^>]+>")


def _clean(s) -> str:
    return _TAG.sub("", str(s or "")).replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&").strip()


class VertexSearchRetriever(RetrieverAdapter):
    def __init__(self, project: str, engine_id: str | None = None, data_store_id: str | None = None,
                 location: str = "global", collection: str = "default_collection",
                 serving_config: str = "default_search", text_field: str | None = None,
                 id_field: str | None = None, source_field: str | None = None, snippets: bool = True,
                 extractive_answers: int = 0, extractive_segments: int = 0, filter: str | None = None,
                 quota_project: str | None = None, api_endpoint: str | None = None,
                 auth: dict | None = None, timeout: float = 30, session=None, **_):
        if bool(engine_id) == bool(data_store_id):
            raise ConfigError("vertex_search: set exactly one of 'engine_id' (search app) or 'data_store_id'")
        self.project, self.location, self.collection = project, location, collection
        self.engine_id, self.data_store_id = engine_id, data_store_id
        host = api_endpoint or ("discoveryengine.googleapis.com" if location == "global"
                                else f"{location}-discoveryengine.googleapis.com")
        self.base = host if host.startswith("http") else f"https://{host}"
        parent = (f"projects/{quote(project, safe='')}/locations/{quote(location, safe='')}"
                  f"/collections/{quote(collection, safe='')}")
        self.store_path = f"{parent}/dataStores/{quote(data_store_id, safe='')}" if data_store_id else None
        owner = f"{parent}/engines/{quote(engine_id, safe='')}" if engine_id else self.store_path
        self.search_url = f"{self.base}/v1/{owner}/servingConfigs/{quote(serving_config, safe='')}:search"
        self.text_field, self.id_field, self.source_field = text_field, id_field, source_field
        self.snippets, self.extractive_answers = bool(snippets), int(extractive_answers)
        self.extractive_segments, self.filter, self.timeout = int(extractive_segments), filter, timeout

        self._owns = session is None
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        if auth:
            self.session.auth = build_auth(auth)
        elif getattr(self.session, "auth", None) is None:
            self.session.auth = GcpServiceAccountAuth()
        if quota_project:
            self.session.headers["x-goog-user-project"] = quota_project

    def _call(self, method: str, url: str, **kw) -> dict:
        r = self.session.request(method, url, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Vertex AI Search {method} {url} failed: HTTP {r.status_code} {r.text[:300]}")
        return r.json()

    def build_body(self, query: str, k: int) -> dict:
        body: dict = {"query": query, "pageSize": k}
        spec: dict = {}
        if self.snippets:
            spec["snippetSpec"] = {"returnSnippet": True}
        ext = {}
        if self.extractive_answers:
            ext["maxExtractiveAnswerCount"] = self.extractive_answers
        if self.extractive_segments:
            ext["maxExtractiveSegmentCount"] = self.extractive_segments
        if ext:
            spec["extractiveContentSpec"] = ext
        if spec:
            body["contentSearchSpec"] = spec
        if self.filter:
            body["filter"] = self.filter
        return body

    def _text(self, struct: dict, derived: dict) -> str:
        if self.text_field and struct.get(self.text_field) is not None:
            return str(struct[self.text_field])
        for key in ("extractive_segments", "extractive_answers"):
            parts = [_clean(x.get("content")) for x in derived.get(key) or [] if isinstance(x, dict)]
            if any(parts):
                return "\n".join(p for p in parts if p)
        snips = [_clean(x.get("snippet")) for x in derived.get("snippets") or []
                 if isinstance(x, dict) and x.get("snippet_status", "SUCCESS") == "SUCCESS"]
        if any(snips):
            return "\n".join(s for s in snips if s)
        return json.dumps(struct, ensure_ascii=False) if struct else ""

    def _to_doc(self, document: dict, rank: int | None = None) -> RetrievedDoc:
        struct = dict(document.get("structData") or {})
        if not struct and document.get("jsonData"):
            try:
                struct = json.loads(document["jsonData"])
            except (TypeError, ValueError):
                struct = {}
        derived = dict(document.get("derivedStructData") or {})
        text = self._text(struct, derived)
        doc_id = struct.get(self.id_field) if self.id_field else None
        meta = {k: v for k, v in struct.items() if k != self.text_field}
        if derived.get("title"):
            meta.setdefault("title", derived["title"])
        if derived.get("link"):
            meta["link"] = derived["link"]
        src = struct.get(self.source_field) if self.source_field else None
        if src is None:
            src = derived.get("link") or (document.get("content") or {}).get("uri")
        meta["source"] = src if src is not None else document.get("id")
        if rank is not None:
            meta["rank"] = rank
        return RetrievedDoc(id=str(doc_id if doc_id is not None else document.get("id", "")),
                            text=text, score=None, metadata=meta)

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        res = self._call("POST", self.search_url, json=self.build_body(query, k))
        out = []
        for i, item in enumerate(res.get("results") or [], start=1):
            document = dict(item.get("document") or {})
            document.setdefault("id", item.get("id"))
            out.append(self._to_doc(document, rank=i))
        return out

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        """Lists documents of the data store (documents.list) and samples from
        a pool of up to 20n. Needs data_store_id. Only documents whose text is
        stored in structData/jsonData are usable; unstructured files (PDFs in
        Cloud Storage) expose no text through this API and are skipped."""
        if not self.store_path:
            raise NotImplementedError("VertexSearchRetriever can sample only with 'data_store_id' "
                                      "(documents.list is a data store API)")
        url = f"{self.base}/v1/{self.store_path}/branches/default_branch/documents"
        pool, token = [], None
        while len(pool) < n * 20:
            params = {"pageSize": 100}
            if token:
                params["pageToken"] = token
            res = self._call("GET", url, params=params)
            pool += res.get("documents") or []
            token = res.get("nextPageToken")
            if not token:
                break
        random.Random(seed).shuffle(pool)
        docs = [d for d in (self._to_doc(x) for x in pool) if d.text]
        if pool and not docs:
            raise NotImplementedError("VertexSearchRetriever cannot sample: the data store's documents carry "
                                      "no text in structData/jsonData (unstructured content is not "
                                      "returned by documents.list)")
        return docs[:n]

    def health_check(self) -> HealthStatus:
        def probe():
            body = self.build_body("health check", 1)
            res = self._call("POST", self.search_url, json=body)
            return f"search returned {len(res.get('results') or [])} result(s)"
        return timed_health(probe)

    def close(self) -> None:
        if self._owns:
            self.session.close()
