"""Load the Northwind docs into a Qdrant collection, to try the Qdrant adapter.

  python examples/load_qdrant.py                              # local file DB, real embeddings
  python examples/load_qdrant.py --embedder hash:256          # instant, no model download
  python examples/load_qdrant.py --url http://localhost:6333  # a Qdrant server (docker run -p 6333:6333 qdrant/qdrant)

Use the same --embedder value in examples/qdrant_retrieval.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient, models  # noqa: E402

from evalkit.embeddings import get_embedder  # noqa: E402
from src.ingest import build_chunks  # noqa: E402


def load(client: QdrantClient, collection: str, embedder: str) -> int:
    chunks = build_chunks()
    embed = get_embedder(embedder)
    vectors = embed([f"{c.title}. {c.text}" for c in chunks])
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config=models.VectorParams(
        size=len(vectors[0]), distance=models.Distance.COSINE))
    client.upsert(collection, points=[
        models.PointStruct(id=i, vector=v, payload={"doc_id": c.id, "text": c.text,
                                                    "source": c.source, "title": c.title})
        for i, (c, v) in enumerate(zip(chunks, vectors))])
    return len(chunks)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="qdrant_data")
    p.add_argument("--url")
    p.add_argument("--collection", default="northwind")
    p.add_argument("--embedder", default="sentence-transformers:all-MiniLM-L6-v2")
    a = p.parse_args()
    client = QdrantClient(url=a.url) if a.url else QdrantClient(path=a.path)
    print(f"Loaded {load(client, a.collection, a.embedder)} chunks into '{a.collection}'")
    client.close()
