"""A simple searchable store built from a folder of .md/.txt files (BM25).
Handy for trying the framework on any documents before you have a real DB.

config:
  type: folder
  path: data/docs
  chunk_words: 80
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..types import RetrievedDoc
from .base import RetrieverAdapter

_STOP = set("a an the is are was were be of to in on for and or with at by from do does how what when "
            "which who can i my it this that as if".split())


def _tok(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in _STOP]


class FolderRetrieverAdapter(RetrieverAdapter):
    def __init__(self, path: str, chunk_words: int = 80, glob: str = "*.*", **_):
        self.docs: list[RetrievedDoc] = []
        for f in sorted(Path(path).rglob(glob)):
            if f.suffix.lower() not in {".md", ".txt"}:
                continue
            words = f.read_text(encoding="utf-8", errors="ignore").split()
            for i in range(0, len(words), chunk_words):
                self.docs.append(RetrievedDoc(id=f"{f.name}#{i // chunk_words}",
                                              text=" ".join(words[i:i + chunk_words]),
                                              metadata={"source": f.name}))
        if not self.docs:
            raise ValueError(f"no .md/.txt files found under {path}")
        self.bm25 = BM25Okapi([_tok(d.text) for d in self.docs])

    def search(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [RetrievedDoc(self.docs[i].id, self.docs[i].text, float(scores[i]), self.docs[i].metadata)
                for i in order]

    def sample(self, n: int, seed: int = 0) -> list[RetrievedDoc]:
        return random.Random(seed).sample(self.docs, min(n, len(self.docs)))
