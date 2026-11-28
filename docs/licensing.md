# Licensing: current state and options

**No licensing change has been made.** The repository is under the owner's
proprietary "all rights reserved" license (`LICENSE`). This page records the
review the roadmap asked for, so the owner can decide.

## Current license and its effect

All rights reserved means nobody else may use, copy, modify or deploy the
code without written permission. That protects the work, but it also blocks
the stated goal of broad adoption: an enterprise cannot install, evaluate or
build plugins for it without a separate agreement.

## Dependency obligations (review)

| Scope | Packages | Licenses | Effect |
|---|---|---|---|
| Core | pyyaml, requests, python-dotenv, pydantic, cryptography, rank-bm25 | MIT, Apache-2.0, BSD-3-Clause, Apache-2.0/BSD | Permissive: compatible with Apache-2.0, proprietary or dual licensing. Keep notices. |
| Server | starlette, uvicorn, httpx, itsdangerous (BSD-3), PyJWT (MIT) | permissive | as above |
| Cloud extras | openai, google-* , boto3 (Apache-2.0); azure-*, anthropic (MIT) | permissive | as above |
| Retriever / framework extras | qdrant-client, pinecone, opentelemetry (Apache-2.0); weaviate-client (BSD-3); langchain-core, llama-index-core, crewai, semantic-kernel, mcp (MIT) | permissive | as above |
| `postgres` / `pgvector` extras | psycopg 3 | **LGPL-3.0** | Fine as an optional, unmodified, separately installed dependency. If a distributed image bundles it, ship its license and allow replacing it (standard pip install satisfies this). Don't vendor or modify it. |

Verify before any release with a scanner (e.g. `pip-licenses` in the release
pipeline); versions change licenses occasionally.

Code ownership: all code in this repository was written for its owner
(including with AI assistance, credited in commit trailers); no third-party
source code is vendored.

## Options

| Model | How it works | Fits when |
|---|---|---|
| **Stay proprietary** (current) | Licenses granted per customer by agreement | Selling it as a closed product or keeping it private |
| **Open core: Apache-2.0 core + commercial extensions** (recommended for adoption) | Engine, CLI, plugin API, OTel ingestion and common integrations under Apache-2.0; enterprise features (SSO dashboard, identity/ACL connectors, monitoring alerts, support) under a commercial license in separate packages | You want enterprises and contributors to adopt it freely and pay for enterprise features. Apache-2.0 adds a patent grant that enterprise legal teams like. |
| **Dual license: AGPL-3.0 + commercial** | Free under AGPL (network use triggers source sharing); companies that won't comply buy a commercial license | You want open source but monetize SaaS / embedded use |
| **Source-available (BSL / Elastic License 2.0)** | Source visible, production or hosted use restricted, often converting to open source after N years | You want visibility but protection against competing hosted offerings |

The code is already structured for open core: the engine has no dependency on
any vendor integration, and every integration is a plugin that can live in a
separate package (entry points), so the line between open and commercial can
be drawn per plugin without touching the engine.

Before changing anything: decide the business model, check any employer IP
agreements, and add a `CONTRIBUTING.md` with a contributor license agreement
(or DCO) if outside contributions will be accepted.
