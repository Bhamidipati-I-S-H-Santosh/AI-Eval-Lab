"""Retriever plugins for hosted search backends (pgvector, Pinecone, Weaviate,
Elasticsearch / OpenSearch, Azure AI Search, Vertex AI Search).

Each module wraps one backend as an evalkit RetrieverAdapter. Vendor SDKs are
imported lazily, only when a config actually builds that retriever without an
injected client. See docs/integrations/retrievers.md."""
