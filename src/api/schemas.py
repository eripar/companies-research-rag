from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    hybrid = "hybrid"
    dense = "dense"
    multi_query = "multi_query"
    hyde = "hyde"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The research question to answer")
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid
    collection_name: str | None = Field(None, description="Override the default Qdrant collection")
    stream: bool = Field(False, description="Stream the response as SSE")


class SourceMetadata(BaseModel):
    source: str | None = None
    filename: str | None = None
    page: int | None = None
    doc_type: str | None = None
    chunk_index: int | None = None
    filing_date: str | None = None
    company: str | None = None

    model_config = {"extra": "allow"}


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceMetadata]
    num_chunks: int
    retrieval_mode: RetrievalMode


class IngestLocalRequest(BaseModel):
    data_dir: str = Field("data/raw", description="Path to directory containing documents")
    strategy: str = Field("recursive", description="Chunking strategy: recursive | sliding_window")
    collection_name: str | None = None


class IngestSECRequest(BaseModel):
    ticker: str = Field(..., description="Company ticker symbol, e.g. AAPL")
    form_type: str = Field("10-K", description="SEC form type: 10-K, 10-Q, 8-K")
    limit: int = Field(5, ge=1, le=20)
    strategy: str = "recursive"
    collection_name: str | None = None


class IngestResponse(BaseModel):
    chunks_upserted: int
    message: str
