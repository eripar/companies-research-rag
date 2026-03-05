"""Integration-style tests for the FastAPI app (mocked LLM and vector store)."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.main import app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestQueryEndpoint:
    @patch("src.api.main.rag_query")
    def test_basic_query(self, mock_rag_query):
        mock_rag_query.return_value = {
            "answer": "Apple's revenue was $383 billion in FY2023.",
            "sources": [
                {
                    "source": "AAPL_10-K_2023.pdf",
                    "filename": "AAPL_10-K_2023.pdf",
                    "page": 42,
                    "doc_type": "10-K",
                    "chunk_index": 5,
                }
            ],
            "num_chunks": 1,
        }

        resp = client.post("/query", json={
            "question": "What was Apple's revenue in FY2023?",
            "retrieval_mode": "hybrid",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "383 billion" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["num_chunks"] == 1

    @patch("src.api.main.rag_query")
    def test_retrieval_modes(self, mock_rag_query):
        mock_rag_query.return_value = {
            "answer": "Test answer.",
            "sources": [],
            "num_chunks": 0,
        }
        for mode in ["hybrid", "dense", "multi_query", "hyde"]:
            resp = client.post("/query", json={
                "question": "Test question?",
                "retrieval_mode": mode,
            })
            assert resp.status_code == 200, f"Failed for mode={mode}"

    def test_invalid_question_too_short(self):
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422
