"""Embed chunks (dense + BM25 sparse), load them into a fresh Qdrant collection, then
point the serving alias at it in one atomic operation. Old collections are dropped.

    python -m ingest.run --corpus data/corpus
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from ingest.chunk import chunk_corpus
from mcp_servers.retrieval.settings import RetrievalSettings


def ingest(
    corpus: Path, settings: RetrievalSettings, batch: int = 64, keep_old: bool = False
) -> str:
    chunks = chunk_corpus(corpus)
    if not chunks:
        raise SystemExit(f"no chunks found under {corpus}; run `python -m ingest.fetch` first")
    print(f"{len(chunks)} chunks from {corpus}")

    dense = TextEmbedding(settings.dense_model)
    sparse = SparseTextEmbedding(settings.sparse_model)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    alias = settings.collection
    name = f"{alias}_{int(time.time())}"
    client.create_collection(
        name,
        vectors_config={
            "dense": models.VectorParams(size=dense.embedding_size, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_payload_index(name, "path_prefixes", models.PayloadSchemaType.KEYWORD)

    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        texts = [f"{c.title}\n{c.section}\n\n{c.text}" for c in part]
        dvecs = list(dense.passage_embed(texts))
        svecs = list(sparse.passage_embed(texts))
        client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=c.id,
                    vector={
                        "dense": d.tolist(),
                        "sparse": models.SparseVector(
                            indices=s.indices.tolist(), values=s.values.tolist()
                        ),
                    },
                    payload=c.payload(),
                )
                for c, d, s in zip(part, dvecs, svecs, strict=True)
            ],
        )
        print(f"  upserted {min(start + batch, len(chunks))}/{len(chunks)}", end="\r")
    print()

    previous = [a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias]
    ops: list[models.CreateAliasOperation | models.DeleteAliasOperation] = []
    if previous:
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=name, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=ops)
    print(f"alias {alias!r} -> {name}")

    if not keep_old:
        for coll in client.get_collections().collections:
            if coll.name.startswith(f"{alias}_") and coll.name != name:
                client.delete_collection(coll.name)
                print(f"dropped {coll.name}")
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--keep-old", action="store_true")
    args = ap.parse_args()
    ingest(args.corpus, RetrievalSettings(), keep_old=args.keep_old)


if __name__ == "__main__":
    main()
