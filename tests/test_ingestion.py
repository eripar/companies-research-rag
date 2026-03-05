"""Tests for document loading and chunking."""

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.chunking import ChunkStrategy, chunk_documents
from src.ingestion.loaders import _infer_doc_type, load_text


def _make_doc(text: str, source: str = "test.txt") -> Document:
    return Document(page_content=text, metadata={"source": source})


class TestDocTypeInference:
    def test_10k(self):
        assert _infer_doc_type("AAPL_10-K_2023.pdf") == "10-K"

    def test_10q(self):
        assert _infer_doc_type("msft_10Q_2024.pdf") == "10-Q"

    def test_earnings(self):
        assert _infer_doc_type("earnings_transcript_q3.txt") == "earnings_transcript"

    def test_fallback(self):
        assert _infer_doc_type("random_document.pdf") == "financial_document"


class TestLoadText:
    def test_loads_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Revenue was $5.2 billion in Q4 2023.\n")
            tmp_path = Path(f.name)

        docs = load_text(tmp_path)
        assert len(docs) == 1
        assert "Revenue" in docs[0].page_content
        assert docs[0].metadata["filename"] == tmp_path.name
        tmp_path.unlink()


class TestChunking:
    def test_recursive_chunk_splits_long_doc(self):
        text = "This is a sentence. " * 200
        doc = _make_doc(text)
        chunks = chunk_documents([doc], strategy=ChunkStrategy.RECURSIVE, chunk_size=256, chunk_overlap=32)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 256 + 50  # some tolerance

    def test_chunk_metadata_enrichment(self):
        doc = _make_doc("Hello world " * 50)
        chunks = chunk_documents([doc], chunk_size=64, chunk_overlap=8)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "char_count" in chunk.metadata
            assert "preview" in chunk.metadata

    def test_sliding_window_strategy(self):
        text = "Financial results for Q3. " * 100
        doc = _make_doc(text)
        chunks = chunk_documents([doc], strategy=ChunkStrategy.SLIDING_WINDOW, chunk_size=128, chunk_overlap=16)
        assert len(chunks) > 1
