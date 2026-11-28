# Installation

## Python package

```bash
pip install "evalkit"                    # core: no vendor SDKs (pyyaml, requests, dotenv, pydantic, cryptography, rank-bm25)
pip install "evalkit[openai]"            # + OpenAI-compatible judges/agents (OpenAI, Groq, vLLM, LiteLLM...)
pip install "evalkit[azure]"             # Azure OpenAI, AI Foundry agents, Key Vault, Blob Storage, Entra ID
pip install "evalkit[aws]"               # Bedrock (models + agents), S3, Secrets Manager, SigV4
pip install "evalkit[gcp]"               # Vertex AI, Cloud Storage, Secret Manager, service accounts
pip install "evalkit[server]"            # REST API, OTLP receiver, dashboard, OIDC SSO
pip install "evalkit[otel]"              # OTLP protobuf ingestion (JSON needs nothing extra)
pip install "evalkit[all]"               # everything above plus qdrant, pinecone, weaviate, postgres, mcp
```

Framework extras: `langchain`, `llamaindex`, `crewai`, `semantic-kernel`.
Retriever extras: `qdrant`, `pgvector`, `pinecone`, `weaviate` (Elasticsearch,
OpenSearch, Azure AI Search and Vertex AI Search use plain REST and need none).

Until the package is published to an index, install from the repository:

```bash
pip install "evalkit[openai] @ git+https://github.com/<owner>/AI-Eval-Lab.git@v0.2.0"
# or from a checkout:
pip install -e ".[dev]"
```

Python 3.10+ is supported. On Windows set `PYTHONUTF8=1` for the demo scripts
in `src/` (evalkit itself reads and writes UTF-8 explicitly).

If a config uses an integration whose extra isn't installed, evalkit stops with:

```
error: plugin 'model:anthropic' needs the package 'anthropic', which is not installed.
Install it with: pip install 'evalkit[anthropic]'
```

## First run

```bash
evalkit plugins list                          # every integration and its verification status
evalkit validate examples/qdrant_retrieval.yaml
python examples/load_qdrant.py --embedder hash:256
evalkit run examples/qdrant_retrieval.yaml    # no API key needed
```

`evalkit` and `python -m evalkit` are the same command.

## Docker

```bash
docker build -t evalkit:0.2.0 --build-arg EXTRAS="server,openai,azure" .
docker run --rm -v $PWD:/work -w /work evalkit:0.2.0 run evals/my-agent.yaml     # CLI
docker run -p 8080:8080 -v $PWD/deploy/examples:/etc/evalkit:ro -v evalkit-data:/data evalkit:0.2.0   # server
```

The image runs as UID 10001, keeps all state under `/data`, works with a
read-only root filesystem, and contains only the extras you chose.

Next: [Configuration](configuration.md), [Deployment](deployment.md).
