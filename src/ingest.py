"""Load markdown documents and split them into chunks.

Two strategies so you can compare them in evals:
  - "section": one chunk per markdown section (## heading)
  - "fixed":   sliding window of N words with overlap
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@dataclass
class Chunk:
    id: str
    source: str      # file name, used to check retrieval against the golden set
    title: str       # document or section title, prepended to the text
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_docs(docs_dir: Path = DOCS_DIR) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(docs_dir.glob("*.md"))}


def _doc_title(text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    return m.group(1).strip() if m else ""


def chunk_by_section(name: str, text: str) -> list[Chunk]:
    doc_title = _doc_title(text)
    parts = re.split(r"^##\s+", text, flags=re.M)[1:]  # drop the H1 preamble
    chunks = []
    for i, part in enumerate(parts):
        heading, _, body = part.partition("\n")
        title = f"{doc_title} > {heading.strip()}"
        chunks.append(Chunk(f"{name}#{i}", name, title, " ".join(body.split())))
    return chunks


def chunk_fixed(name: str, text: str, size: int = 60, overlap: int = 15) -> list[Chunk]:
    doc_title = _doc_title(text)
    body = re.sub(r"^#+\s+.*$", "", text, flags=re.M)  # strip headings
    words = body.split()
    step = max(1, size - overlap)
    chunks = []
    for i, start in enumerate(range(0, max(1, len(words)), step)):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append(Chunk(f"{name}#{i}", name, doc_title, " ".join(piece)))
        if start + size >= len(words):
            break
    return chunks


ATTACKS_DIR = DOCS_DIR.parent / "attacks"


def build_chunks(strategy: str = "section", size: int = 60, overlap: int = 15,
                 include_attacks: bool = False) -> list[Chunk]:
    """include_attacks=True mixes in poisoned documents for security testing."""
    docs = load_docs()
    if include_attacks:
        docs.update(load_docs(ATTACKS_DIR))
    chunks: list[Chunk] = []
    for name, text in docs.items():
        if strategy == "section":
            chunks += chunk_by_section(name, text)
        elif strategy == "fixed":
            chunks += chunk_fixed(name, text, size, overlap)
        else:
            raise ValueError(f"unknown chunking strategy: {strategy}")
    return chunks


if __name__ == "__main__":
    for c in build_chunks():
        print(f"{c.id:28} {c.title}")
