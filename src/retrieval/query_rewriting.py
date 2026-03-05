"""
Query rewriting strategies to improve retrieval recall.

Strategies:
- multi_query: Generate N alternative phrasings of the query, retrieve for each,
               deduplicate, return union. Good for broadening recall.
- hyde: Hypothetical Document Embeddings — generate a hypothetical answer, embed it,
        use that embedding for retrieval. Good for precision on factual queries.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.vector_store import get_vector_store


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=512,
        temperature=0.3,
        api_key=settings.anthropic_api_key,
    )


# ---------------------------------------------------------------------------
# Multi-query expansion
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert at financial research. Generate {n} alternative phrasings "
        "of the following query that would help retrieve relevant information from SEC filings, "
        "earnings transcripts, and financial reports. "
        "Return ONLY the queries, one per line, no numbering or explanation."
    )),
    ("human", "{query}"),
])


def multi_query_retrieve(
    query: str,
    n_variants: int = 3,
    top_k_final: int | None = None,
    collection_name: str | None = None,
) -> list[Document]:
    """
    Generate n query variants, retrieve for each, return deduplicated union.
    """
    k = top_k_final or settings.top_k_final
    llm = _get_llm()
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()

    variants_text = chain.invoke({"query": query, "n": n_variants})
    variants = [q.strip() for q in variants_text.strip().split("\n") if q.strip()]
    all_queries = [query] + variants[:n_variants]

    seen_ids: set[str] = set()
    all_docs: list[Document] = []
    store = get_vector_store(collection_name=collection_name)

    for q in all_queries:
        results = store.similarity_search(q, k=k)
        for doc in results:
            doc_id = _stable_id(doc)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    # Trim to top_k_final
    return all_docs[:k * 2]


# ---------------------------------------------------------------------------
# HyDE (Hypothetical Document Embeddings)
# ---------------------------------------------------------------------------

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a financial analyst. Write a concise, factual paragraph that would "
        "appear in a 10-K or earnings call transcript and directly answers the question below. "
        "Be specific and use financial terminology. Do not mention that this is hypothetical."
    )),
    ("human", "{query}"),
])


def hyde_retrieve(
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
) -> list[Document]:
    """
    Generate a hypothetical document for the query, embed it, retrieve by similarity.
    """
    k = top_k or settings.top_k_final
    llm = _get_llm()
    chain = HYDE_PROMPT | llm | StrOutputParser()

    hypothetical_doc = chain.invoke({"query": query})

    store = get_vector_store(collection_name=collection_name)
    return store.similarity_search(hypothetical_doc, k=k)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_id(doc: Document) -> str:
    src = doc.metadata.get("source", "")
    idx = doc.metadata.get("chunk_index", "")
    return f"{src}::{idx}" if src else str(hash(doc.page_content[:200]))
