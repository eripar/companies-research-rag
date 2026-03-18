# Fintech RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system over financial documents — SEC filings, earnings call transcripts, and financial reports.

## Features

- **Document ingestion**: Loads and parses SEC filings (10-K, 10-Q, 8-K) and earnings call transcripts
- **Chunking strategies**: Recursive, semantic, and sliding-window chunking with configurable overlap
- **Vector storage**: Qdrant (default) with Pinecone support via abstraction layer
- **Hybrid search**: BM25 sparse + dense vector retrieval with RRF fusion
- **Query rewriting**: HyDE (Hypothetical Document Embeddings) and multi-query expansion
- **RAG chain**: LangChain LCEL pipeline with citation tracking
- **FastAPI**: REST API with streaming support

## Stack

| Layer | Tech |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector DB | Qdrant (local or cloud) |
| Sparse search | BM25 via `rank_bm25` |
| LLM | Claude claude-sonnet-4-6 via Anthropic SDK |
| API | FastAPI + Uvicorn |
| Data | SEC EDGAR API, local PDF/TXT |

## Project Structure

```
companies-research-rag/
├── src/
│   ├── ingestion/      # Document loading, chunking, ingestion pipeline
│   ├── retrieval/      # Embeddings, vector store, hybrid search, query rewriting
│   ├── generation/     # Prompts, RAG chain
│   └── api/            # FastAPI app, schemas, routes
├── data/
│   └── raw/            # Drop PDFs / transcripts here
├── notebooks/          # Exploration and prototyping
└── tests/
```

## Quickstart

**Requirements:** Python 3.11+, Docker

```bash
# Install dependencies (use a Python 3.11 environment)
pip install -e ".[dev]"

# Set environment variables
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, and SEC_CONTACT_EMAIL (your real email)

# Start Qdrant locally (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Start API
uvicorn src.api.main:app --reload
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ingest/local` | Ingest documents from a local directory |
| `POST` | `/ingest/sec` | Fetch and ingest SEC filings from EDGAR by ticker |
| `POST` | `/query` | Query the RAG pipeline |

Interactive API docs available at `http://localhost:8000/docs` when running locally.

### Example: Ingest a SEC filing

```bash
curl -X POST http://localhost:8000/ingest/sec \
  -H "Content-Type: application/json" \
  -d '{"ticker": "CRWD", "form_type": "10-K", "limit": 1}'
```

### Example: Query

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are CrowdStrike'\''s main revenue streams?"}' | python3 -m json.tool
```

## SEC EDGAR Access

No registration is required, but SEC EDGAR requires a valid email address in the `User-Agent` header. Set `SEC_CONTACT_EMAIL` in your `.env` to your real email — requests using placeholder emails may be blocked with a 503 error.
