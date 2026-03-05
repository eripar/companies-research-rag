"""
Document loaders for financial documents.

Supports:
- PDF (local files — SEC filings, earnings transcripts)
- SEC EDGAR API (10-K, 10-Q, 8-K by CIK/ticker)
- Plain text
"""

import re
from pathlib import Path
from typing import Iterator

import httpx
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Local file loaders
# ---------------------------------------------------------------------------

def load_pdf(file_path: Path) -> list[Document]:
    """Load a PDF and return one Document per page."""
    import pdfplumber

    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "page": i + 1,
                        "total_pages": len(pdf.pages),
                        "doc_type": _infer_doc_type(file_path.name),
                    },
                ))
    return docs


def load_text(file_path: Path) -> list[Document]:
    """Load a plain-text file as a single Document."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return [Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "filename": file_path.name,
            "doc_type": _infer_doc_type(file_path.name),
        },
    )]


def load_directory(data_dir: Path, glob: str = "**/*") -> list[Document]:
    """Recursively load all supported files from a directory."""
    loaders = {".pdf": load_pdf, ".txt": load_text}
    docs = []
    for path in sorted(data_dir.glob(glob)):
        if path.is_file() and path.suffix.lower() in loaders:
            docs.extend(loaders[path.suffix.lower()](path))
    return docs


# ---------------------------------------------------------------------------
# SEC EDGAR loader
# ---------------------------------------------------------------------------

EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"

HEADERS = {"User-Agent": "companies-research-rag contact@example.com"}


def fetch_sec_filing(cik: int, accession_number: str) -> list[Document]:
    """
    Fetch a single SEC filing by CIK and accession number.

    Args:
        cik: Company CIK (int).
        accession_number: e.g. '0000950170-23-035122'

    Returns:
        List of Documents (one per section/page).
    """
    accession_nodash = accession_number.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}-index.htm"

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        index_resp = client.get(index_url)
        index_resp.raise_for_status()

        # Find the primary document link
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(index_resp.text, "html.parser")
        doc_link = None
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 4 and "10-" in cells[3].get_text():
                doc_link = cells[2].find("a")
                break
        if doc_link is None:
            raise ValueError(f"Could not find primary document in {index_url}")

        filing_url = f"https://www.sec.gov{doc_link['href']}"
        filing_resp = client.get(filing_url)
        filing_resp.raise_for_status()

        text = BeautifulSoup(filing_resp.text, "html.parser").get_text(separator="\n")
        text = _clean_sec_text(text)

    return [Document(
        page_content=text,
        metadata={
            "source": filing_url,
            "cik": cik,
            "accession_number": accession_number,
            "doc_type": "sec_filing",
        },
    )]


def iter_recent_filings(ticker: str, form_type: str = "10-K", limit: int = 5) -> Iterator[dict]:
    """
    Yield metadata dicts for recent filings from EDGAR for a given ticker.

    Args:
        ticker: e.g. 'AAPL'
        form_type: '10-K', '10-Q', '8-K', etc.
        limit: max number of filings to return
    """
    # Resolve ticker -> CIK
    with httpx.Client(headers=HEADERS) as client:
        resp = client.get("https://www.sec.gov/files/company_tickers.json")
        resp.raise_for_status()
        tickers_data = resp.json()

    cik = None
    for entry in tickers_data.values():
        if entry["ticker"].upper() == ticker.upper():
            cik = entry["cik_str"]
            break
    if cik is None:
        raise ValueError(f"Ticker {ticker!r} not found in EDGAR")

    with httpx.Client(headers=HEADERS) as client:
        resp = client.get(EDGAR_SUBMISSIONS_URL.format(cik=int(cik)))
        resp.raise_for_status()
        submissions = resp.json()

    filings = submissions.get("filings", {}).get("recent", {})
    count = 0
    for i, form in enumerate(filings.get("form", [])):
        if form == form_type and count < limit:
            yield {
                "cik": int(cik),
                "accession_number": filings["accessionNumber"][i],
                "filing_date": filings["filingDate"][i],
                "form_type": form,
                "company": submissions.get("name", ticker),
            }
            count += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_doc_type(filename: str) -> str:
    name = filename.lower()
    if "10-k" in name or "10k" in name:
        return "10-K"
    if "10-q" in name or "10q" in name:
        return "10-Q"
    if "8-k" in name or "8k" in name:
        return "8-K"
    if "transcript" in name or "earnings" in name:
        return "earnings_transcript"
    return "financial_document"


def _clean_sec_text(text: str) -> str:
    """Remove boilerplate whitespace and artifacts from SEC filings."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
