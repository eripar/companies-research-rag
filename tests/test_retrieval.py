"""Tests for hybrid search and query rewriting (mocked)."""

import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch

from src.retrieval.hybrid_search import _bm25_search, _reciprocal_rank_fusion, _doc_id


def _make_doc(content: str, source: str = "test.txt", idx: int = 0) -> Document:
    return Document(page_content=content, metadata={"source": source, "chunk_index": idx})


class TestBM25Search:
    def test_returns_top_k(self):
        docs = [
            _make_doc("revenue profit earnings income", idx=0),
            _make_doc("risk factors regulatory compliance", idx=1),
            _make_doc("revenue growth year over year", idx=2),
        ]
        results = _bm25_search("revenue", docs, k=2)
        assert len(results) == 2
        # Both revenue docs should outscore the risk doc
        result_contents = [r.page_content for r in results]
        assert all("revenue" in c for c in result_contents)

    def test_empty_corpus(self):
        results = _bm25_search("query", [], k=5)
        assert results == []


class TestRRFFusion:
    def test_basic_fusion(self):
        list1 = [_make_doc("a", idx=0), _make_doc("b", idx=1), _make_doc("c", idx=2)]
        list2 = [_make_doc("b", idx=1), _make_doc("a", idx=0), _make_doc("d", idx=3)]
        fused = _reciprocal_rank_fusion([list1, list2], k=3)
        assert len(fused) == 3
        # 'a' and 'b' appear in both lists, should rank above 'c' and 'd'
        top_ids = {_doc_id(d) for d in fused[:2]}
        assert _doc_id(list1[0]) in top_ids  # 'a'
        assert _doc_id(list1[1]) in top_ids  # 'b'

    def test_deduplication(self):
        doc = _make_doc("same doc", idx=0)
        list1 = [doc]
        list2 = [doc]
        fused = _reciprocal_rank_fusion([list1, list2], k=5)
        assert len(fused) == 1

    def test_respects_k(self):
        docs = [_make_doc(f"doc {i}", idx=i) for i in range(10)]
        fused = _reciprocal_rank_fusion([docs], k=3)
        assert len(fused) == 3
