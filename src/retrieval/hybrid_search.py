"""
Hybrid search: dense vector retrieval + BM25 sparse retrieval, fused with RRF.

Reciprocal Rank Fusion (RRF) is a simple, parameter-free rank fusion method
that outperforms score normalization in most retrieval benchmarks.

Reference: Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods"
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document

from src.config import settings
from src.retrieval.vector_store import get_vector_store


def hybrid_search(
    query: str,
    top_k_dense: int | None = None,
    top_k_sparse: int | None = None,
    top_k_final: int | None = None,
    collection_name: str | None = None,
    corpus: list[Document] | None = None,
) -> list[Document]:
    """
    Run hybrid search over the vector store + an optional in-memory BM25 corpus.

    Args:
        query: User query string.
        top_k_dense: Number of dense candidates to retrieve.
        top_k_sparse: Number of BM25 candidates to retrieve.
        top_k_final: Number of results after RRF fusion.
        collection_name: Qdrant collection to search.
        corpus: If provided, run BM25 over this corpus instead of the vector store docs.
                Useful for re-ranking an already-retrieved set.

    Returns:
        Top-k documents ranked by RRF score.
    """
    k_dense = top_k_dense or settings.top_k_dense
    k_sparse = top_k_sparse or settings.top_k_sparse
    k_final = top_k_final or settings.top_k_final

    store = get_vector_store(collection_name=collection_name)

    # Dense retrieval
    dense_results = store.similarity_search(query, k=k_dense)

    # Sparse (BM25) retrieval
    bm25_corpus = corpus if corpus is not None else dense_results
    sparse_results = _bm25_search(query, bm25_corpus, k=k_sparse)

    # RRF fusion
    fused = _reciprocal_rank_fusion([dense_results, sparse_results], k=k_final)
    return fused


def _bm25_search(query: str, docs: list[Document], k: int) -> list[Document]:
    """Run BM25 over a list of Documents and return top-k."""
    from rank_bm25 import BM25Okapi

    if not docs:
        return []

    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    ranked_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked_indices[:k]]


def _reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 5,
    rrf_k: int = 60,
) -> list[Document]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: Each sublist is a ranked list of Documents.
        k: Number of documents to return.
        rrf_k: RRF smoothing constant (60 is standard).
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            # Use page_content hash as a stable key
            doc_id = _doc_id(doc)
            scores[doc_id] += 1.0 / (rrf_k + rank)
            doc_map[doc_id] = doc

    ranked = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)
    return [doc_map[did] for did in ranked[:k]]


def _doc_id(doc: Document) -> str:
    """Stable identifier for a document chunk."""
    src = doc.metadata.get("source", "")
    idx = doc.metadata.get("chunk_index", "")
    # Fall back to content hash if metadata is sparse
    return f"{src}::{idx}" if src else str(hash(doc.page_content[:200]))
