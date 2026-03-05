# Fintech RAG Pipeline: Technical White Paper

**Version 0.1 — March 2026**

---

## Abstract

This document describes the architecture, operational mechanics, and design rationale of a production-grade Retrieval-Augmented Generation (RAG) system built for financial document analysis. The system enables analysts and researchers to query large corpora of SEC filings, earnings call transcripts, and financial reports using natural language — receiving precise, cited answers grounded in the source documents rather than model hallucination. This white paper covers the business motivation, a step-by-step account of how the system processes a query from human input to generated answer, and the key architectural decisions made during its design.

---

## Section 1: Business Case

### The Problem with LLMs in Finance

Large language models are powerful reasoning engines, but they have a fundamental limitation in financial contexts: their knowledge is frozen at a training cutoff. Markets move daily. Earnings are reported quarterly. Material events — acquisitions, restatements, regulatory actions — can change the financial picture of a company overnight. An LLM asked about a company's current leverage ratio or its most recent guidance will either hallucinate a plausible-sounding answer or correctly admit it doesn't know. Neither outcome is useful to a professional making decisions.

Beyond recency, financial analysis demands precision that general-purpose LLMs are not calibrated for. The difference between operating income and net income, between diluted and basic EPS, between a 10-K and a 10-Q — these distinctions matter enormously and LLMs frequently blur them when reasoning from parametric memory alone.

### Why RAG Solves This

Retrieval-Augmented Generation addresses both problems by separating the knowledge store from the reasoning engine. Rather than asking the model to remember facts, the system retrieves the relevant passages from a curated document corpus at query time and provides them as context to the model. The model's job shifts from recollection to comprehension and synthesis — a task it performs well.

For financial use cases, this architecture delivers:

- **Recency**: The knowledge base reflects whatever documents have been ingested. A 10-Q filed yesterday can be queryable within minutes of ingestion.
- **Precision**: Answers are grounded in exact text from source documents, with citations traceable to specific filings and page numbers.
- **Auditability**: Every answer can be traced back to its source chunks. This is a compliance requirement in many financial workflows.
- **Scope control**: The system only answers from what it has been given. Hallucination risk is bounded to the retrieval and generation steps, not unconstrained parametric recall.

### Target Users

This system is designed for financial analysts, portfolio managers, research teams, and AI-augmented workflows that require reliable, document-grounded answers from large collections of financial filings. It is equally applicable to internal enterprise document corpora (deal memos, credit reports, research notes) and public regulatory filings.

### Return on Investment

The ROI case for this system rests on analyst time. A typical sell-side analyst spends several hours manually reviewing a single 10-K for key data points. A RAG system that can surface the relevant passages in seconds — with citations — compresses that workflow dramatically. At scale across a portfolio of companies and a team of analysts, the time savings compound quickly. The secondary benefit is consistency: every analyst queries the same corpus with the same retrieval logic, reducing the variance introduced by individual reading habits and document selection.

---

## Section 2: How the System Works

### The Human Interface

The system exposes two primary interfaces for human interaction:

**REST API** — A FastAPI server provides HTTP endpoints that any front-end, workflow tool, or downstream application can call. The primary endpoint is `POST /query`, which accepts a JSON body containing the question and retrieval configuration. Responses are returned as JSON with the answer text, source metadata, and chunk count. Streaming is supported via Server-Sent Events for real-time token-by-token output in chat-style interfaces.

**CLI (Ingestion)** — Document ingestion is controlled via a command-line interface. An operator runs `python -m src.ingestion.pipeline local --data-dir data/raw/` to process local files, or `python -m src.ingestion.pipeline sec --ticker AAPL --form 10-K --limit 5` to pull directly from the SEC EDGAR API.

**Exploration Notebook** — A Jupyter notebook (`notebooks/exploration.ipynb`) provides an interactive layer for prototyping queries, inspecting retrieved chunks, and experimenting with retrieval modes before committing to production configurations.

---

### Step-by-Step: Ingestion Path

This path runs once (or periodically) to populate the knowledge base.

**Step 1 — Document Loading**

The operator points the CLI at a data source. For local files, the loader (`src/ingestion/loaders.py`) walks the target directory and dispatches each file to the appropriate parser: `pdfplumber` for PDFs (preserving page-level structure) and a UTF-8 text reader for plain text. For EDGAR ingestion, the loader first resolves the ticker to a CIK using the SEC's public company ticker JSON, then fetches the filing index page, identifies the primary document, downloads the HTML, and strips it to plain text using BeautifulSoup. In both cases, the output is a list of LangChain `Document` objects, each carrying its text content and a metadata dictionary that records the source path, filename, inferred document type, and page number where applicable.

**Step 2 — Chunking**

Raw documents are too large to embed meaningfully in a single vector and too large to fit in a model's context window as retrieved context. The chunking module (`src/ingestion/chunking.py`) splits each document into smaller passages. The default strategy is recursive character splitting: the splitter attempts to divide text at the largest available semantic boundary first (triple newlines, then double newlines, then single newlines, then sentence boundaries, then words) and only falls back to smaller boundaries when a chunk would exceed the configured size limit (default: 1,024 characters with 128-character overlap between adjacent chunks). After splitting, each chunk receives enriched metadata: a sequential index within its source document, a character count, and a 120-character preview for debugging and display purposes.

**Step 3 — Embedding**

Each chunk's text is sent to OpenAI's `text-embedding-3-large` model, which returns a 3,072-dimensional float vector representing the semantic content of that passage. The embedding model is initialized once and cached for the lifetime of the process to avoid redundant API round-trips.

**Step 4 — Upsert to Qdrant**

The chunk text, its embedding vector, and its metadata are written to the Qdrant vector database. Qdrant stores these in a collection (default name: `fintech_docs`) using a cosine-similarity HNSW index. If the collection does not yet exist, it is created automatically with the correct vector dimensions. The pipeline logs progress via a rich console progress bar and reports the total number of chunks upserted on completion.

---

### Step-by-Step: Query Path

This path runs on every user question.

**Step 1 — API receives the request**

The user submits a `POST /query` request with a JSON body, for example:

```json
{
  "question": "What was Apple's gross margin trend from 2021 to 2023?",
  "retrieval_mode": "hybrid",
  "stream": false
}
```

FastAPI validates the request against the `QueryRequest` Pydantic schema — enforcing that the question is at least 3 characters, that the retrieval mode is one of the four valid options, and that all types are correct. Invalid requests are rejected with a 422 before any retrieval occurs.

**Step 2 — Query Rewriting (optional, mode-dependent)**

Depending on the `retrieval_mode` selected, the query may be rewritten before retrieval:

- **`hybrid`** and **`dense`**: The original query is used as-is.
- **`multi_query`**: The system prompts Claude to generate three alternative phrasings of the question. For example, "What was Apple's gross margin trend from 2021 to 2023?" might yield variants like "How did Apple's profitability as measured by gross margin change over three years?" and "Apple gross margin FY2021 FY2022 FY2023 comparison." All variants are retrieved against separately and the results are deduplicated.
- **`hyde`** (Hypothetical Document Embeddings): The system prompts Claude to write a short paragraph that *would appear in a 10-K* and directly answer the question — as if the answer already existed in the corpus. This hypothetical passage is then embedded and used as the retrieval query. Because the hypothetical answer looks like source document text rather than a user question, it retrieves semantically closer matches.

**Step 3 — Retrieval**

The retrieval module (`src/retrieval/hybrid_search.py`) runs dense and sparse retrieval in sequence:

*Dense retrieval*: The query (or hypothetical document) is embedded and a k-nearest-neighbor search is run against the Qdrant HNSW index. The top 10 candidates by cosine similarity are returned.

*Sparse retrieval (BM25)*: The BM25Okapi algorithm scores each of the dense candidates by term frequency, treating the dense result set as the corpus. This surfaces candidates that are lexically relevant to the query terms — important for financial queries that contain specific ticker symbols, accounting line items, or regulatory terminology that may not map cleanly into embedding space.

*RRF Fusion*: The dense and sparse ranked lists are merged using Reciprocal Rank Fusion. Each document receives a score of `1 / (60 + rank)` from each list in which it appears, and scores are summed. Documents appearing near the top of both lists receive the highest fused scores. The top 5 documents by fused score are selected as the context for generation.

**Step 4 — Context Formatting**

The retrieved chunks are formatted into a structured context string (`src/generation/prompts.py`). Each chunk is labeled with its source document and page number and separated by a horizontal rule, making it easy for the model to distinguish between passages from different documents.

**Step 5 — Generation**

The formatted context and the original question are inserted into a system prompt via LangChain's LCEL chain. The system prompt instructs Claude to answer using only the provided context, to cite specific figures and document sections, and to format a source list at the end of its response. Claude (`claude-sonnet-4-6`) generates the answer. If `stream: true` was requested, tokens are emitted as Server-Sent Events as they are produced; otherwise the complete response is collected and returned as JSON.

**Step 6 — Response**

The API returns a `QueryResponse` object containing the answer text, an array of source metadata objects (one per retrieved chunk), the number of chunks used, and the retrieval mode that was applied. The caller can use the source metadata to link back to the original documents.

---

## Section 3: Design Decisions and Tradeoffs

### Vector Database: Qdrant over Pinecone or Chroma

The selection of Qdrant as the vector store was driven by three requirements: local development without external dependencies, native hybrid search support, and production-grade payload filtering.

Pinecone was ruled out for local development friction — it is cloud-only and requires account provisioning before a single vector can be written. For a portfolio project and for teams who want to iterate without cloud costs during development, this is a significant constraint. Chroma was ruled out because it lacks native sparse vector support and is not considered production-grade for high-volume deployments.

Weaviate was the closest alternative. It supports hybrid search natively and can run locally. The decision came down to API ergonomics and LangChain integration quality — Qdrant's `langchain-qdrant` package is more actively maintained and the payload filtering API is more explicit. Qdrant also supports multiple quantization schemes (scalar, product, binary) natively, which matters when the corpus grows to millions of chunks.

The tradeoff accepted: Qdrant requires running a Docker container locally, whereas Chroma requires no infrastructure at all. For early prototyping, Chroma would be simpler. The architecture is abstracted through the `get_vector_store()` function, making a future swap possible without touching retrieval or generation code.

### Retrieval Fusion: RRF over Score Normalization

Two standard approaches exist for fusing dense and sparse ranked lists: score normalization (converting both score distributions to [0,1] and weighting them) and Reciprocal Rank Fusion (combining based on rank position only).

Score normalization was rejected because BM25 scores are not bounded — they scale with corpus size and document length, making normalization across different queries and corpus states unstable. An `alpha` weight tuned on one query set will generalize poorly to others. RRF sidesteps this entirely by ignoring raw scores. The smoothing constant `k=60` is the empirically validated standard from the original Cormack et al. paper and generalizes well without query-specific tuning.

The tradeoff: RRF discards score magnitude information. Two documents at rank 1 and rank 2 receive similar fusion scores even if the rank-1 document was retrieved with dramatically higher confidence. In practice, rank order is more stable and reliable than raw score magnitude across embedding models and BM25 implementations, so this loss is acceptable.

### Chunking: Recursive over Fixed-Size or Semantic

Three chunking strategies were considered: fixed-size sliding window, recursive character splitting, and semantic chunking (splitting at detected topic shifts using embedding similarity).

Semantic chunking was rejected for this version. It requires embedding every candidate split point, making ingestion significantly slower and more expensive. It also requires threshold tuning per corpus type. For a first version, the complexity cost is not justified by the marginal recall improvement.

Fixed-size sliding window chunks are simple and ensure uniform coverage but frequently split mid-sentence or mid-paragraph, degrading chunk coherence and retrieval quality.

Recursive character splitting was selected because it respects the natural hierarchy of financial document structure — major section breaks, paragraph breaks, sentence boundaries — and only falls back to smaller units when necessary. Both strategies are implemented and can be selected at ingestion time; the recursive default can be overridden per ingestion run without code changes.

The accepted tradeoff: recursive splitting produces chunks of variable length, which means some chunks will be short (a paragraph break in a dense section) and some will approach the maximum size. This variance in chunk size is preferable to consistently incoherent fixed-size chunks for retrieval quality.

### Embedding Model: text-embedding-3-large over Alternatives

OpenAI's `text-embedding-3-large` at 3,072 dimensions was selected over `text-embedding-3-small` (1,536 dimensions) and open-source alternatives such as BGE-large or E5-large.

`text-embedding-3-large` demonstrates meaningfully better retrieval performance on financial and legal benchmarks. The higher dimensionality preserves finer-grained semantic distinctions that matter in technical domains. The cost difference between small and large is approximately 3x per token — significant at scale but manageable for a corpus of SEC filings.

Open-source alternatives were not selected for this version because they require local GPU inference infrastructure to run at reasonable speed. The architecture accepts this as a vendor dependency with a mitigation: the embedding model is isolated behind the `get_embeddings()` function and any LangChain-compatible embeddings class can be substituted without touching retrieval or ingestion logic.

A relevant nuance of `text-embedding-3`: dimensions can be truncated without re-embedding the entire corpus. A 3,072-dimensional vector can be truncated to 1,536 dimensions with acceptable quality loss if storage cost becomes a constraint at scale.

### Generation Model: Claude over GPT-4

Claude (`claude-sonnet-4-6`) was selected as the generation model. For financial RAG, the critical model property is instruction-following fidelity: the model must answer *from the provided context only* and resist the temptation to supplement with parametric knowledge. Claude's instruction adherence is well-documented in enterprise financial use cases and its extended context window (up to 200K tokens) provides headroom for future experiments with larger retrieved context sets.

The tradeoff is API vendor concentration — both the embedding model (OpenAI) and the generation model introduce external API dependencies. A fully on-premise alternative would use an open-source embedding model and a locally-hosted LLM, at the cost of infrastructure complexity and inference speed. The abstraction boundaries in the codebase (embedding behind `get_embeddings()`, LLM instantiation behind `_get_llm()`) keep this swap localized if required.

### Query Rewriting: Four Modes over a Single Strategy

Rather than committing to one retrieval strategy, four modes were implemented and exposed as a runtime parameter: `hybrid`, `dense`, `multi_query`, and `hyde`.

This reflects a practical reality: no single retrieval strategy dominates across all query types. Factual lookup queries ("What was net income in Q3 2024?") are well-served by hybrid search. Broad conceptual queries ("What are the key risks in the semiconductor supply chain?") benefit from multi-query expansion. Queries where the user knows what an answer *looks like* but not the exact terminology perform best with HyDE.

The cost of this decision is complexity — four code paths to test and maintain. The benefit is that the system can be tuned per deployment context without architectural changes, and the exploration notebook makes it straightforward to compare retrieval modes on the same query before selecting a default for a given corpus.

---

## Conclusion

The fintech RAG pipeline described here addresses a genuine gap in how LLMs are applied in financial contexts: grounding analytical queries in authoritative, timestamped source documents rather than parametric model memory. The architecture prioritizes retrieval quality (hybrid search, multiple query rewriting modes), operational practicality (local-first development, CLI ingestion, typed API), and maintainability (clear abstraction boundaries that allow component-level substitution as requirements evolve). The design decisions documented in Section 3 reflect deliberate tradeoffs rather than defaults — each layer of the stack was selected for a concrete reason, and the alternatives considered remain viable paths as the system scales.
