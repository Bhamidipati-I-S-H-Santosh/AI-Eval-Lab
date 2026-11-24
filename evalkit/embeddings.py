"""Embedding functions, chosen by a string in the config.

  hash:256                          built-in hashing embedder; no download, for CI/tests only
  sentence-transformers:<model>     local and free, e.g. sentence-transformers:all-MiniLM-L6-v2
  openai:<model>                    any OpenAI-compatible /embeddings endpoint
                                    (uses EMBED_BASE_URL / EMBED_API_KEY, falling back to LLM_*)
  my_module:my_fn                   your own callable: list[str] -> list[list[float]]

Use the SAME embedder the system under test used to build its index.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Callable

Embedder = Callable[[list[str]], list[list[float]]]


def hash_embedder(dim: int = 256) -> Embedder:
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * dim
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
    return embed


def get_embedder(spec: str | Callable) -> Embedder:
    if callable(spec):
        return spec
    kind, _, arg = spec.partition(":")
    if kind == "hash":
        return hash_embedder(int(arg or 256))
    if kind == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(arg or "all-MiniLM-L6-v2")
        return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()
    if kind == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("EMBED_BASE_URL") or os.getenv("LLM_BASE_URL"),
                        api_key=os.getenv("EMBED_API_KEY") or os.getenv("LLM_API_KEY"))
        return lambda texts: [d.embedding for d in client.embeddings.create(model=arg, input=texts).data]
    from .utils import import_object
    return import_object(spec)
