"""
RAG chain — ties together retrieval and generation.

Usage:
    from src.generation.chain import rag_query, rag_stream

    result = rag_query("What was Apple's revenue in FY2023?")
    print(result["answer"])
    print(result["sources"])
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import settings
from src.generation.prompts import RAG_PROMPT, format_context
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.query_rewriting import hyde_retrieve, multi_query_retrieve


def _get_llm(streaming: bool = False) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        streaming=streaming,
        api_key=settings.anthropic_api_key,
    )


def _retrieve(
    query: str,
    retrieval_mode: str = "hybrid",
    collection_name: str | None = None,
) -> list[Document]:
    """
    Retrieve relevant chunks for a query.

    Args:
        retrieval_mode: One of 'hybrid', 'dense', 'multi_query', 'hyde'.
    """
    if retrieval_mode == "hybrid":
        return hybrid_search(query, collection_name=collection_name)
    elif retrieval_mode == "dense":
        from src.retrieval.vector_store import get_vector_store
        store = get_vector_store(collection_name=collection_name)
        return store.similarity_search(query, k=settings.top_k_final)
    elif retrieval_mode == "multi_query":
        return multi_query_retrieve(query, collection_name=collection_name)
    elif retrieval_mode == "hyde":
        return hyde_retrieve(query, collection_name=collection_name)
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode!r}")


def rag_query(
    question: str,
    retrieval_mode: str = "hybrid",
    collection_name: str | None = None,
) -> dict:
    """
    Run a full RAG query and return the answer with source metadata.

    Returns:
        {
            "answer": str,
            "sources": list[dict],  # metadata from retrieved chunks
            "num_chunks": int,
        }
    """
    docs = _retrieve(question, retrieval_mode=retrieval_mode, collection_name=collection_name)
    context = format_context(docs)

    llm = _get_llm(streaming=False)
    chain = RAG_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [doc.metadata for doc in docs],
        "num_chunks": len(docs),
    }


def rag_stream(
    question: str,
    retrieval_mode: str = "hybrid",
    collection_name: str | None = None,
) -> Iterator[str]:
    """
    Stream the RAG answer token by token.

    Usage:
        for token in rag_stream("What was NVDA's gross margin in Q3 2024?"):
            print(token, end="", flush=True)
    """
    docs = _retrieve(question, retrieval_mode=retrieval_mode, collection_name=collection_name)
    context = format_context(docs)

    llm = _get_llm(streaming=True)
    chain = RAG_PROMPT | llm | StrOutputParser()

    yield from chain.stream({"context": context, "question": question})


async def rag_astream(
    question: str,
    retrieval_mode: str = "hybrid",
    collection_name: str | None = None,
) -> AsyncIterator[str]:
    """Async streaming variant for use in FastAPI SSE endpoints."""
    docs = _retrieve(question, retrieval_mode=retrieval_mode, collection_name=collection_name)
    context = format_context(docs)

    llm = _get_llm(streaming=True)
    chain = RAG_PROMPT | llm | StrOutputParser()

    async for token in chain.astream({"context": context, "question": question}):
        yield token
