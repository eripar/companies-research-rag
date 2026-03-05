"""
Qdrant vector store abstraction.

get_vector_store() returns a LangChain QdrantVectorStore that can be used
directly as a retriever or passed to the hybrid search module.
"""

from functools import lru_cache

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings
from src.retrieval.embeddings import get_embeddings


def _get_client() -> QdrantClient:
    if settings.qdrant_api_key:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Create the collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )


@lru_cache(maxsize=8)
def get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    """
    Return a cached QdrantVectorStore for the given collection.

    Args:
        collection_name: Overrides settings.qdrant_collection_name when provided.
    """
    name = collection_name or settings.qdrant_collection_name
    client = _get_client()
    ensure_collection(client, name)
    return QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=get_embeddings(),
    )
