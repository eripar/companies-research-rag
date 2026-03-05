"""
End-to-end ingestion pipeline: load -> chunk -> embed -> upsert.

CLI usage:
    python -m src.ingestion.pipeline --data-dir data/raw/
    python -m src.ingestion.pipeline --ticker AAPL --form 10-K --limit 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import track

from src.ingestion.chunking import ChunkStrategy, chunk_documents
from src.ingestion.loaders import fetch_sec_filing, iter_recent_filings, load_directory

console = Console()


def ingest_local(
    data_dir: Path,
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    collection_name: str | None = None,
) -> int:
    """
    Load all supported files from data_dir, chunk them, and upsert to vector store.

    Returns:
        Number of chunks upserted.
    """
    from src.retrieval.vector_store import get_vector_store

    console.print(f"[bold]Loading documents from[/bold] {data_dir}")
    docs = load_directory(data_dir)
    if not docs:
        console.print("[yellow]No documents found.[/yellow]")
        return 0
    console.print(f"Loaded {len(docs)} raw documents")

    chunks = chunk_documents(docs, strategy=strategy)
    console.print(f"Created {len(chunks)} chunks (strategy={strategy})")

    store = get_vector_store(collection_name=collection_name)
    ids = store.add_documents(list(track(chunks, description="Upserting...")))
    console.print(f"[green]Done.[/green] Upserted {len(ids)} chunks.")
    return len(ids)


def ingest_sec(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 5,
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    collection_name: str | None = None,
) -> int:
    """
    Fetch recent SEC filings for a ticker from EDGAR, chunk, and upsert.

    Returns:
        Number of chunks upserted.
    """
    from src.retrieval.vector_store import get_vector_store

    console.print(f"[bold]Fetching {limit}x {form_type} for {ticker}[/bold]")
    all_chunks = []
    for filing_meta in iter_recent_filings(ticker, form_type=form_type, limit=limit):
        console.print(f"  Fetching {filing_meta['accession_number']} ({filing_meta['filing_date']})")
        try:
            docs = fetch_sec_filing(filing_meta["cik"], filing_meta["accession_number"])
            for doc in docs:
                doc.metadata.update(filing_meta)
            chunks = chunk_documents(docs, strategy=strategy)
            all_chunks.extend(chunks)
            console.print(f"    -> {len(chunks)} chunks")
        except Exception as exc:
            console.print(f"    [red]Error:[/red] {exc}")

    if not all_chunks:
        console.print("[yellow]No chunks produced.[/yellow]")
        return 0

    store = get_vector_store(collection_name=collection_name)
    ids = store.add_documents(list(track(all_chunks, description="Upserting...")))
    console.print(f"[green]Done.[/green] Upserted {len(ids)} chunks.")
    return len(ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fintech RAG ingestion pipeline")
    sub = p.add_subparsers(dest="command")

    local = sub.add_parser("local", help="Ingest local files")
    local.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    local.add_argument("--strategy", choices=[s.value for s in ChunkStrategy], default="recursive")
    local.add_argument("--collection", default=None)

    sec = sub.add_parser("sec", help="Ingest from SEC EDGAR")
    sec.add_argument("--ticker", required=True)
    sec.add_argument("--form", default="10-K")
    sec.add_argument("--limit", type=int, default=5)
    sec.add_argument("--strategy", choices=[s.value for s in ChunkStrategy], default="recursive")
    sec.add_argument("--collection", default=None)

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "local":
        ingest_local(
            data_dir=args.data_dir,
            strategy=ChunkStrategy(args.strategy),
            collection_name=args.collection,
        )
    elif args.command == "sec":
        ingest_sec(
            ticker=args.ticker,
            form_type=args.form,
            limit=args.limit,
            strategy=ChunkStrategy(args.strategy),
            collection_name=args.collection,
        )
    else:
        parser.print_help()
