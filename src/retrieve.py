"""Retrieval: BM25 (keyword), vector (embeddings), hybrid (both, merged with
Reciprocal Rank Fusion), plus an optional cross-encoder reranker.

Vector search and reranking need `sentence-transformers` (free, runs locally).
BM25 only needs `rank-bm25`, which keeps CI fast.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .ingest import Chunk

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_STOP = set(
    "a an the is are was were be been of to in on for and or with at by from "
    "do does did i my me can how what when which who much many it this that "
    "there as per if".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in _STOP]


class Retriever:
    def __init__(self, chunks: list[Chunk], method: str = "bm25", rerank: bool = False):
        self.chunks = chunks
        self.method = method
        self.rerank = rerank
        self._texts = [f"{c.title}. {c.text}" for c in chunks]
        self._bm25 = BM25Okapi([tokenize(t) for t in self._texts])
        self._embedder = None
        self._emb = None
        self._reranker = None
        if method in ("vector", "hybrid"):
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
            self._emb = self._embedder.encode(self._texts, normalize_embeddings=True)
        if rerank:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL)

    # each ranker returns chunk indices, best first
    def _rank_bm25(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        return sorted(range(len(scores)), key=lambda i: -scores[i])

    def _rank_vector(self, query: str) -> list[int]:
        q = self._embedder.encode([query], normalize_embeddings=True)[0]
        scores = self._emb @ q
        return sorted(range(len(scores)), key=lambda i: -scores[i])

    @staticmethod
    def _rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused, key=lambda i: -fused[i])

    def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        if self.method == "bm25":
            order = self._rank_bm25(query)
        elif self.method == "vector":
            order = self._rank_vector(query)
        elif self.method == "hybrid":
            order = self._rrf([self._rank_bm25(query), self._rank_vector(query)])
        else:
            raise ValueError(f"unknown retrieval method: {self.method}")

        if self._reranker is not None:
            pool = order[: max(top_k * 4, 10)]  # rerank a wider candidate pool
            scores = self._reranker.predict([(query, self._texts[i]) for i in pool])
            order = [i for _, i in sorted(zip(scores, pool), key=lambda x: -x[0])]

        return [self.chunks[i] for i in order[:top_k]]
