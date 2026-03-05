"""
FastAPI application for the Fintech RAG pipeline.

Endpoints:
    GET  /health          - Health check
    POST /ingest/local    - Ingest documents from a local directory
    POST /ingest/sec      - Ingest SEC filings from EDGAR
    POST /query           - Query the RAG pipeline (JSON or SSE streaming)
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    IngestLocalRequest,
    IngestResponse,
    IngestSECRequest,
    QueryRequest,
    QueryResponse,
    SourceMetadata,
)
from src.generation.chain import rag_astream, rag_query

app = FastAPI(
    title="Fintech RAG API",
    description="Production RAG pipeline over SEC filings and earnings transcripts",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.post("/ingest/local", response_model=IngestResponse)
async def ingest_local(req: IngestLocalRequest):
    from src.ingestion.chunking import ChunkStrategy
    from src.ingestion.pipeline import ingest_local as _ingest_local

    data_dir = Path(req.data_dir)
    if not data_dir.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {data_dir}")

    try:
        strategy = ChunkStrategy(req.strategy)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy!r}")

    n = _ingest_local(data_dir, strategy=strategy, collection_name=req.collection_name)
    return IngestResponse(chunks_upserted=n, message=f"Ingested {n} chunks from {data_dir}")


@app.post("/ingest/sec", response_model=IngestResponse)
async def ingest_sec(req: IngestSECRequest):
    from src.ingestion.chunking import ChunkStrategy
    from src.ingestion.pipeline import ingest_sec as _ingest_sec

    try:
        strategy = ChunkStrategy(req.strategy)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy!r}")

    try:
        n = _ingest_sec(
            ticker=req.ticker,
            form_type=req.form_type,
            limit=req.limit,
            strategy=strategy,
            collection_name=req.collection_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return IngestResponse(
        chunks_upserted=n,
        message=f"Ingested {n} chunks for {req.ticker} {req.form_type}",
    )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if req.stream:
        # SSE streaming — return as text/event-stream
        async def event_stream():
            async for token in rag_astream(
                req.question,
                retrieval_mode=req.retrieval_mode.value,
                collection_name=req.collection_name,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = rag_query(
        req.question,
        retrieval_mode=req.retrieval_mode.value,
        collection_name=req.collection_name,
    )
    return QueryResponse(
        answer=result["answer"],
        sources=[SourceMetadata(**s) for s in result["sources"]],
        num_chunks=result["num_chunks"],
        retrieval_mode=req.retrieval_mode,
    )
