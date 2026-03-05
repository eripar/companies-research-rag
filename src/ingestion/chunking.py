"""
Chunking strategies for financial documents.

Strategies:
- recursive: LangChain RecursiveCharacterTextSplitter (default)
- semantic: Splits on semantic similarity shifts (experimental)
- sliding_window: Fixed-size chunks with configurable overlap
"""

from enum import Enum

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"


def chunk_documents(
    docs: list[Document],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into chunks using the specified strategy.

    Args:
        docs: Raw documents from loaders.
        strategy: Chunking strategy to use.
        chunk_size: Override default from settings.
        chunk_overlap: Override default from settings.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if strategy == ChunkStrategy.RECURSIVE:
        chunks = _recursive_chunk(docs, size, overlap)
    elif strategy == ChunkStrategy.SLIDING_WINDOW:
        chunks = _sliding_window_chunk(docs, size, overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return _enrich_chunk_metadata(chunks)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _recursive_chunk(docs: list[Document], size: int, overlap: int) -> list[Document]:
    """
    Recursive character splitting — respects paragraph and sentence boundaries.
    Financial documents benefit from splitting on section headers too.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=[
            "\n\n\n",   # major section breaks
            "\n\n",     # paragraph breaks
            "\n",       # line breaks
            ". ",       # sentence boundaries
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)


def _sliding_window_chunk(docs: list[Document], size: int, overlap: int) -> list[Document]:
    """Fixed-size sliding window — good for uniform dense retrieval coverage."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def _enrich_chunk_metadata(chunks: list[Document]) -> list[Document]:
    """Add chunk index, character count, and preview to each chunk's metadata."""
    source_counter: dict[str, int] = {}

    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = source_counter.get(src, 0)
        source_counter[src] = idx + 1

        chunk.metadata["chunk_index"] = idx
        chunk.metadata["char_count"] = len(chunk.page_content)
        chunk.metadata["preview"] = chunk.page_content[:120].replace("\n", " ")

    return chunks
