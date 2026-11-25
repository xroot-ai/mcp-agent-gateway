"""Retrieval server settings. Backend URL and secrets live here, server-side only: the
gateway and the model can never point a search at a different backend."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection: str = "k8s_docs"  # an alias; ingest swaps it atomically
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    max_passage_chars: int = 1200
    default_top_k: int = 5
    token: str = ""  # bearer token required on HTTP
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8101
