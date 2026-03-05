"""
Prompt templates for the RAG chain.
"""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a financial research analyst assistant with deep expertise in SEC filings, \
earnings call transcripts, and financial reports.

Answer the user's question using ONLY the context provided below. \
If the context does not contain enough information to answer confidently, say so explicitly \
rather than speculating.

Rules:
- Be precise and cite specific figures, dates, and document sections when available.
- Format numbers clearly (e.g., "$2.3 billion", "15.4% YoY").
- If multiple documents are relevant, synthesize across them.
- At the end of your answer, list the sources you used in this format:
  Sources: [filename, page X] or [company 10-K, filing date]

Context:
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_context(docs) -> str:
    """Format retrieved documents into the context string."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source_label = (
            meta.get("filename")
            or meta.get("accession_number")
            or meta.get("source", f"Document {i}")
        )
        page = meta.get("page", "")
        page_str = f", page {page}" if page else ""
        parts.append(f"[{i}] ({source_label}{page_str})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
