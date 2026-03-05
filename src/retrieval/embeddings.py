"""
Embedding model setup.

Uses OpenAI text-embedding-3-large by default.
Swap to any LangChain-compatible embeddings class without touching retrieval logic.
"""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from src.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return a cached embeddings instance."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.openai_api_key,
    )
