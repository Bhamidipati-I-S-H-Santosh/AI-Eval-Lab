# Retriever integrations

A retriever plugin wraps a search backend so evalkit can:

- score retrieval quality on its own (`RetrieverAsAgent`: hit rate, source recall, MRR...),
- generate test questions from sampled documents (`sample()`),
- wrap the backend with the prompt-injection "poisoned retriever" to test how the agent handles hostile content.

Each plugin returns `RetrievedDoc(id, text, score, metadata)`. When the backend stores the logical
source (file name, URL, document id), the plugin puts it in `metadata["source"]`; the
`source_recall` metric matches a test case's `expected_sources` against `doc.id`, an `"id#"` prefix
of `doc.id`, or `metadata["source"]`.

**Data flow.** evalkit sends queries straight from where it runs to *your* search service, using
credentials you supply. No documents or queries pass through any third party. When a vector mode
embeds the query locally (`embedder:`), the embedding model runs where evalkit runs (or calls the
embedding endpoint you configured).

**Embedders.** Vector modes embed the query with `embedder:` (see `evalkit/embeddings.py`:
`sentence-transformers:<model>`, `openai:<model>`, `module:function`, `hash:<dim>` for tests). It
**must** be the same model, with the same dimensions, that built the index. Modes that let the
service vectorize (Weaviate `near_text`, Azure integrated vectorization) need no embedder.

**Secrets.** Every string option accepts `${ENV_VAR}` and `${secret:name}` references.

**Dependencies.** Vendor SDKs are imported only when a config builds that retriever without an
injected client. A missing SDK raises `PluginDependencyError` with the `pip install` command.

| type | backend | dependency | search modes | `sample()` |
|---|---|---|---|---|
| `pgvector` | PostgreSQL + pgvector | `psycopg` (3) | vector | yes |
| `pinecone` | Pinecone | `pinecone` (v5+) | vector | serverless indexes |
| `weaviate` | Weaviate | `weaviate-client` (v4) | near_vector, near_text, hybrid, bm25 | yes |
| `elasticsearch` / `opensearch` | Elasticsearch, OpenSearch, AWS OpenSearch | `requests` | bm25, knn, hybrid (RRF) | yes |
| `azure_ai_search` | Azure AI Search | `requests` | keyword, vector, hybrid, semantic | yes |
| `vertex_search` | Vertex AI Search (Discovery Engine) | `requests` (+ `google-auth` for default auth) | engine ranking | with `data_store_id` |

Every plugin implements `health_check()` (run by `python -m evalkit plugins health CONFIG`) with a cheap probe,
and `close()`. Injected clients/sessions/connections are left open for their owner to close.

---

## PostgreSQL + pgvector (`pgvector`)

Status: mock-tested (implemented against the documented API; not yet verified against a live service)

```yaml
retriever:
  type: pgvector
  dsn: ${PG_DSN}                    # postgresql://user:pass@host:5432/db  (sslmode=require etc. in the DSN)
  table: public.chunks              # optionally schema-qualified
  text_column: content
  id_column: id
  embedding_column: embedding
  metadata_columns: [title, url]    # returned in doc.metadata
  source_column: url                # copied to metadata["source"]
  distance: cosine                  # cosine (<=>) | l2 (<->) | inner_product (<#>)
  filter: { tenant: acme, lang: [en, de] }   # col = value / col = ANY(list) / null -> IS NULL
  embedder: sentence-transformers:all-MiniLM-L6-v2
```

- **Auth:** whatever the DSN carries (password, `sslmode`, client certs). A read-only role with
  `SELECT` on the table is enough.
- **Search:** `ORDER BY embedding <op> %s::vector LIMIT k`, with the query vector, filter values and
  `k` sent as bind parameters. Pick `distance` to match the operator class of your vector index
  (`vector_cosine_ops` -> cosine, `vector_l2_ops` -> l2, `vector_ip_ops` -> inner_product) or
  Postgres will not use the index.
- **Score** (higher is better): cosine `1 - distance`, inner_product the inner product, l2 `-distance`.
- **Identifiers** (table, columns, filter keys) must match `^[A-Za-z_][A-Za-z0-9_]{0,62}$` and are
  double-quoted, so they are case-sensitive. Anything else is rejected with a `ConfigError`.
- **sample():** yes. `setseed(seed)` then `ORDER BY random() LIMIT n` (full scan; fine for
  evaluation-sized tables).
- **health_check:** selects one row of the id, text and embedding columns.

## Pinecone (`pinecone`)

Status: mock-tested (implemented against the documented API; not yet verified against a live service)

```yaml
retriever:
  type: pinecone
  api_key: ${PINECONE_API_KEY}      # falls back to the PINECONE_API_KEY env var
  index_name: company-docs          # or  host: company-docs-abc123.svc.us-east-1.pinecone.io
  namespace: prod                   # optional
  text_field: text                  # metadata key holding the chunk text
  id_field: null                    # metadata key used as doc id (default: vector id)
  source_field: source              # metadata key copied to metadata["source"]
  filter: { department: { $eq: finance } }   # Pinecone metadata filter, passed through as-is
  embedder: sentence-transformers:all-MiniLM-L6-v2
```

- **Auth:** Pinecone API key.
- **Search:** `index.query(vector, top_k, include_metadata=True, namespace, filter)`; score is
  Pinecone's similarity score.
- **sample():** `index.list()` + `index.fetch()`; Pinecone supports listing ids on **serverless**
  indexes only (pod-based indexes raise `NotImplementedError`). Ignores `filter`.
- **health_check:** `describe_index_stats()` (total vector count).

## Weaviate (`weaviate`)

Status: mock-tested (implemented against the documented API; not yet verified against a live service)

```yaml
retriever:
  type: weaviate
  collection: CompanyDocs
  # self-hosted -> weaviate.connect_to_custom
  url: http://localhost:8080
  grpc_port: 50051                  # grpc_host defaults to the url's host
  # Weaviate Cloud -> weaviate.connect_to_weaviate_cloud
  # cluster_url: https://my-cluster.weaviate.cloud
  api_key: ${WEAVIATE_API_KEY}      # optional for self-hosted
  headers: { X-OpenAI-Api-Key: ${OPENAI_API_KEY} }   # for server-side vectorizer modules
  mode: hybrid                      # near_vector | near_text | hybrid | bm25
  alpha: 0.5                        # hybrid: 1 = pure vector, 0 = pure keyword
  embedder: null                    # required for near_vector; optional for hybrid
  text_field: text
  id_field: null                    # default: object uuid
  source_field: source
  target_vector: null               # named vectors
  tenant: null                      # multi-tenancy
  filter: { department: finance }   # Filter.by_property(k).equal(v), ANDed
```

- **Auth:** Weaviate API key (`Auth.api_key`), plus any vectorizer headers.
- **Modes:** `near_vector` embeds locally; `near_text` and `hybrid` without an embedder use the
  collection's vectorizer; `bm25` is keyword only.
- **Score:** `1 - distance` for near_vector / near_text; Weaviate's fused score for hybrid / bm25.
- **sample():** yes. `fetch_objects(limit=20n, filters)` then a seeded shuffle.
- **health_check:** `aggregate.over_all(total_count=True)`.

## Elasticsearch / OpenSearch (`elasticsearch`, `opensearch`)

Status: mock-tested (implemented against the documented API; not yet verified against a live service)

Plain REST via `requests` (no SDK), so the same plugin serves Elastic Cloud, self-managed
Elasticsearch, OpenSearch and Amazon OpenSearch Service.

```yaml
retriever:
  type: elasticsearch               # or opensearch (same plugin)
  flavor: elasticsearch             # elasticsearch | opensearch: selects the knn syntax
  url: https://my-deployment.es.us-east-1.aws.elastic.cloud:443
  index: company-docs               # comma lists and wildcards allowed
  mode: hybrid                      # bm25 | knn | hybrid
  text_field: content
  fields: [title^2, content]        # bm25 multi_match fields (default: [text_field])
  id_field: null                    # _source field used as id (default: _id)
  source_field: source
  vector_field: embedding           # knn / hybrid
  embedder: sentence-transformers:all-MiniLM-L6-v2
  num_candidates: 100               # Elasticsearch knn
  hybrid_window: 50                 # hits fetched per leg before fusion
  rrf_k: 60
  filter: { department: finance, lang: [en, de] }   # term / terms filters
  verify_tls: true                  # or a CA bundle path
  timeout: 30
  auth:
    type: api_key
    key: ${ES_API_KEY}
    header: Authorization
    prefix: "ApiKey "
```

- **Auth options:**
  - Elastic API key: as above.
  - Basic auth: `username: elastic`, `password: ${ES_PASSWORD}`.
  - Bearer token: `auth: { type: bearer, token: ${TOKEN} }`.
  - Amazon OpenSearch Service (IAM): `auth: { type: aws_sigv4, service: es, region: us-east-1 }`
    (`service: aoss` for OpenSearch Serverless). Credentials come from the standard AWS chain.
- **Modes:**
  - `bm25`: `multi_match` over `fields`, with filters in `bool.filter`.
  - `knn`: Elasticsearch uses the top-level `knn` clause (`field`, `query_vector`, `k`,
    `num_candidates`, `filter`); OpenSearch uses the `knn` query
    (`{"knn": {field: {"vector", "k", "filter"}}}`; the inner `filter` needs the lucene or faiss engine).
  - `hybrid`: runs bm25 and knn separately and fuses them client-side with reciprocal rank fusion
    (`score = sum 1/(rrf_k + rank)`), so it works on every license tier and on OpenSearch. The
    returned score is the RRF score.
- The vector field is excluded from `_source` in responses.
- **sample():** yes. `function_score` + `random_score {seed, field: _seq_no}`: a server-side,
  reproducible random sample.
- **health_check:** `GET /{index}/_count`.

## Azure AI Search (`azure_ai_search`)

Status: mock-tested (implemented against the documented API; not yet verified against a live service)

`POST {endpoint}/indexes/{index}/docs/search?api-version=2024-07-01` via `requests`.

```yaml
retriever:
  type: azure_ai_search
  endpoint: https://my-search.search.windows.net
  index: company-docs
  api_key: ${AZURE_SEARCH_KEY}      # query key is enough
  mode: hybrid                      # keyword | vector | hybrid | semantic
  vector_fields: [contentVector]    # vector / hybrid (optional in semantic -> semantic hybrid)
  embedder: null                    # set -> vectorQueries kind "vector" (local embedding)
                                    # unset -> kind "text" (index's integrated vectorizer)
  semantic_configuration: default   # semantic mode
  text_field: content
  id_field: id
  source_field: metadata_storage_name
  select: [id, content, title, metadata_storage_name]
  search_fields: [content, title]
  filter: { category: finance }     # -> "category eq 'finance'"; or a raw OData string
  api_version: 2024-07-01
```

- **Auth options:**
  - `api_key:` sent as the `api-key` header.
  - Microsoft Entra ID: `auth: { type: entra_id }` (uses `DefaultAzureCredential`; the scope
    defaults to `https://search.azure.com/.default`). The identity needs the **Search Index Data
    Reader** role and the service must have role-based access enabled.
  - Any other evalkit auth block (e.g. `oauth2_client_credentials` with that scope).
- **Modes:** `keyword` (`search`), `vector` (`vectorQueries` only), `hybrid` (both; Azure fuses
  them with RRF), `semantic` (`queryType: semantic`, `semanticConfiguration`, extractive captions;
  adds `vectorQueries` too when `vector_fields` is set).
- **Score:** `@search.rerankerScore` when present (semantic), else `@search.score`. Captions go to
  `metadata["captions"]`; vector fields and `@search.*` keys are dropped from metadata.
- **Filters:** dict values are escaped as OData literals (`'` doubled); field names are validated.
- **sample():** yes. `search: "*"` for up to 20n docs (max 1000), then a seeded shuffle.
- **health_check:** `GET /indexes/{index}/docs/$count`.

## Vertex AI Search (`vertex_search`)

Status: mock-tested (implemented against the documented API; not yet verified against a live service)

`POST https://discoveryengine.googleapis.com/v1/projects/{project}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_search:search`
(or `.../dataStores/{data_store_id}/servingConfigs/default_search:search`).

```yaml
retriever:
  type: vertex_search
  project: my-gcp-project
  location: global                  # global | us | eu (us/eu use {location}-discoveryengine.googleapis.com)
  engine_id: company-docs-app       # or data_store_id: company-docs-ds
  serving_config: default_search
  text_field: null                  # structData field with the text (structured data stores)
  id_field: null                    # default: document id
  source_field: null                # default: derivedStructData.link (gs:// or web URL), then document id
  snippets: true
  extractive_answers: 0             # maxExtractiveAnswerCount (Enterprise edition)
  extractive_segments: 0            # maxExtractiveSegmentCount (Enterprise edition)
  filter: 'category: ANY("finance")'
  quota_project: null               # x-goog-user-project, needed with end-user ADC credentials
  auth: { type: gcp_service_account }   # default when no auth/session is given
```

- **Auth:** Google credentials via `GcpServiceAccountAuth` by default (service-account key in
  `GOOGLE_APPLICATION_CREDENTIALS`, workload identity, or `gcloud auth application-default login`).
  The identity needs `discoveryengine.servingConfigs.search` (e.g. role **Discovery Engine Viewer**).
- **Search:** `pageSize = k`. Text comes from, in order: `text_field` in `structData`, extractive
  segments, extractive answers, snippets (HTML stripped), else the JSON of `structData`. For
  unstructured data stores, turn on `extractive_segments` (Enterprise) to get full passages;
  snippets are short.
- **Score:** the API returns no relevance score by default, so `score` is `None`; order is Vertex's
  ranking and `metadata["rank"]` is 1-based.
- **sample():** only with `data_store_id` (`documents.list` on `branches/default_branch`). Only
  documents carrying text in `structData`/`jsonData` are usable; unstructured files expose no text
  through that API, so a store with only those raises `NotImplementedError`.
- **health_check:** a 1-result search.

---

## Testing a plugin without the service

Every plugin accepts an injected backend object so it can be tested offline:
`connection=` (pgvector), `index=` (Pinecone), `client=` (Weaviate), `session=` (Elasticsearch,
Azure AI Search, Vertex AI Search). See `tests/test_retrievers.py` for fakes that mimic each
service's response shapes.

```python
from evalkit.plugins import REGISTRY
r = REGISTRY.build("retriever", {"type": "pinecone", "index_name": "docs", "embedder": "hash:256",
                                 "index": my_fake_index})
```
