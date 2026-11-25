"""Async hybrid search over Qdrant: dense + sparse (BM25) prefetch fused with RRF.

Returns trimmed passages only: no vectors, ids or raw payload leave the server.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import AsyncQdrantClient, models

from mcp_servers.retrieval.overrides import SearchOverrides
from mcp_servers.retrieval.settings import RetrievalSettings


@dataclass
class Passage:
    title: str
    url: str
    section: str
    text: str
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "url": self.url,
            "section": self.section,
            "text": self.text,
            "score": round(self.score, 4),
        }


class Searcher(Protocol):
    async def search(self, query: str, top_k: int, overrides: SearchOverrides) -> list[Passage]: ...


def trim(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[: end + 1] if end > limit * 0.6 else cut).rstrip() + " …"


class QdrantHybridSearcher:
    def __init__(self, settings: RetrievalSettings) -> None:
        self._s = settings
        self._client: AsyncQdrantClient | None = None
        self._dense = None
        self._sparse = None
        self._lock = asyncio.Lock()

    async def warm_up(self) -> None:
        async with self._lock:
            if self._client is None:
                self._client = AsyncQdrantClient(
                    url=self._s.qdrant_url, api_key=self._s.qdrant_api_key
                )
            if self._dense is None:
                self._dense, self._sparse = await asyncio.to_thread(self._load_models)

    def _load_models(self):  # noqa: ANN202
        from fastembed import SparseTextEmbedding, TextEmbedding

        return TextEmbedding(self._s.dense_model), SparseTextEmbedding(self._s.sparse_model)

    def _embed(self, query: str) -> tuple[list[float], models.SparseVector]:
        assert self._dense is not None and self._sparse is not None
        dense = next(iter(self._dense.query_embed(query)))
        sparse = next(iter(self._sparse.query_embed(query)))
        return dense.tolist(), models.SparseVector(
            indices=sparse.indices.tolist(), values=sparse.values.tolist()
        )

    async def search(self, query: str, top_k: int, overrides: SearchOverrides) -> list[Passage]:
        await self.warm_up()
        assert self._client is not None
        dense, sparse = await asyncio.to_thread(self._embed, query)
        flt = None
        if overrides.section_prefix:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="path_prefixes", match=models.MatchValue(value=overrides.section_prefix)
                    )
                ]
            )
        result = await self._client.query_points(
            collection_name=self._s.collection,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=top_k * 4, filter=flt),
                models.Prefetch(query=sparse, using="sparse", limit=top_k * 4, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=["title", "url", "section", "text"],
        )
        passages = []
        for point in result.points:
            if overrides.min_score is not None and point.score < overrides.min_score:
                continue
            p = point.payload or {}
            passages.append(
                Passage(
                    title=str(p.get("title", "")),
                    url=str(p.get("url", "")),
                    section=str(p.get("section", "")),
                    text=trim(str(p.get("text", "")), self._s.max_passage_chars),
                    score=float(point.score),
                )
            )
        return passages
