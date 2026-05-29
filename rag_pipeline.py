"""
LedgerAI - RAG Pipeline
=============================================
Stack:
  - PyMuPDF        → PDF text extraction
  - sentence-transformers → Embeddings
  - ChromaDB       → Vector store
  - Google Gemini  → LLM (Free!)

Install:
  pip install pymupdf chromadb sentence-transformers google-generativeai fastapi uvicorn python-multipart python-dotenv
"""

import os
import json
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-key-here")
genai.configure(api_key=GEMINI_API_KEY)
CHROMA_PATH       = "./chroma_db"
COLLECTION_NAME   = "ledger_invoices"
EMBED_MODEL       = "all-MiniLM-L6-v2"   # Fast, free, good quality
CHUNK_SIZE        = 500                   # characters per chunk
CHUNK_OVERLAP     = 50
TOP_K             = 5                     # how many chunks to retrieve


# ─────────────────────────────────────────
# 1. PDF EXTRACTION
# ─────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes, filename: str) -> str:
    """Extract all text from a PDF invoice."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"[Page {page_num + 1}]\n{text.strip()}")

    doc.close()
    combined = f"[Invoice File: {filename}]\n\n" + "\n\n".join(full_text)
    return combined


# ─────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start  = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        last_period = chunk.rfind(". ")
        if last_period > chunk_size // 2:
            chunk = chunk[: last_period + 1]
            end   = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 20]   # drop tiny fragments


# ─────────────────────────────────────────
# 3. VECTOR STORE (ChromaDB)
# ─────────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedder   = SentenceTransformer(EMBED_MODEL)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_invoice(self, text: str, filename: str, invoice_id: str):
        """Chunk and embed an invoice into the vector store."""
        chunks = chunk_text(text)

        ids        = [f"{invoice_id}_chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.encode(chunks).tolist()
        metadatas  = [{"filename": filename, "invoice_id": invoice_id, "chunk_index": i}
                      for i in range(len(chunks))]

        self.collection.upsert(
            ids        = ids,
            embeddings = embeddings,
            documents  = chunks,
            metadatas  = metadatas,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve most relevant chunks for a query."""
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings = query_embedding,
            n_results        = top_k,
            include          = ["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "content":    doc,
                "filename":   meta["filename"],
                "invoice_id": meta["invoice_id"],
                "score":      round(1 - dist, 3),   # cosine similarity
            })

        return hits

    def delete_invoice(self, invoice_id: str):
        """Remove all chunks for an invoice."""
        existing = self.collection.get(where={"invoice_id": invoice_id})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

    def list_invoices(self) -> list[str]:
        """Return unique invoice filenames stored."""
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return list({m["filename"] for m in all_meta})

    def count(self) -> int:
        return self.collection.count()


# ─────────────────────────────────────────
# 4. RAG QUERY ENGINE (Gemini)
# ─────────────────────────────────────────
class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vs    = vector_store
        self.model = genai.GenerativeModel("gemini-1.5-flash")  # Free tier model

    def build_context(self, hits: list[dict]) -> str:
        """Format retrieved chunks into a context block."""
        parts = []
        for i, hit in enumerate(hits, 1):
            parts.append(
                f"[Source {i} | File: {hit['filename']} | Relevance: {hit['score']}]\n"
                f"{hit['content']}"
            )
        return "\n\n---\n\n".join(parts)

    def query(self, user_question: str, invoice_data: Optional[dict] = None) -> dict:
        """
        Full RAG pipeline:
          1. Retrieve relevant chunks
          2. Build prompt with context
          3. Call Gemini
          4. Return answer + sources
        """
        # Step 1 — Retrieve
        hits    = self.vs.search(user_question, top_k=TOP_K)
        context = self.build_context(hits)

        # Step 2 — Build structured data context
        structured = ""
        if invoice_data:
            structured = f"\n\nStructured Invoice Database Summary:\n{json.dumps(invoice_data, indent=2)}"

        # Step 3 — Build full prompt for Gemini
        full_prompt = f"""You are LedgerAI's intelligent finance assistant.
You have access to the user's invoice documents through a RAG system.

Your job:
- Answer questions about invoices, vendors, spending, and financial patterns
- Be precise with numbers and dates
- Always cite which invoice/file your answer comes from
- If you cannot find the answer in the provided context, say so clearly
- Keep answers concise but complete
- Flag any anomalies or concerns you notice

Question: {user_question}

Retrieved Invoice Context:
{context}
{structured}

Please answer based on the above context. Cite sources where relevant."""

        # Step 4 — Call Gemini
        response = self.model.generate_content(full_prompt)
        answer   = response.text

        return {
            "answer":      answer,
            "sources":     [{"filename": h["filename"], "score": h["score"]} for h in hits],
            "chunks_used": len(hits),
        }

    def explain_anomaly(self, invoice: dict) -> str:
        """Use Gemini to explain why an invoice was flagged."""
        prompt = f"""You are a financial auditor AI. Explain in 2-3 sentences why this invoice looks suspicious:

Invoice details:
{json.dumps(invoice, indent=2)}

Focus on: unusual amounts, missing vendor info, tax inconsistencies, or pattern deviations.
Be specific and actionable."""

        response = self.model.generate_content(prompt)
        return response.text

    def generate_report(self, invoice_data: dict) -> str:
        """Auto-generate a professional financial summary report."""
        prompt = f"""Generate a professional financial report for the following invoice data.
Structure it with: Executive Summary, Spending Analysis, Vendor Overview, Risk Flags, and Recommendations.

Data:
{json.dumps(invoice_data, indent=2)}

Write in a professional tone suitable for a CFO or finance manager."""

        response = self.model.generate_content(prompt)
        return response.text


# ─────────────────────────────────────────
# 5. FASTAPI ROUTES
# ─────────────────────────────────────────
app   = FastAPI(title="LedgerAI RAG API")
vs    = VectorStore()
engine = RAGEngine(vs)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


class QueryRequest(BaseModel):
    question:     str
    invoice_data: Optional[dict] = None


class AnomalyRequest(BaseModel):
    invoice: dict


class ReportRequest(BaseModel):
    invoice_data: dict


@app.post("/api/rag/upload")
async def upload_invoice(file: UploadFile = File(...)):
    """Upload and index a PDF invoice into the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    pdf_bytes  = await file.read()
    invoice_id = Path(file.filename).stem
    text       = extract_text_from_pdf(pdf_bytes, file.filename)
    n_chunks   = vs.add_invoice(text, file.filename, invoice_id)

    return {
        "message":    f"Indexed {file.filename}",
        "chunks":     n_chunks,
        "invoice_id": invoice_id,
        "total_docs": vs.count(),
    }


@app.post("/api/rag/query")
async def query_invoices(req: QueryRequest):
    """Ask a question about your invoices using RAG + Claude."""
    if vs.count() == 0:
        return {
            "answer":  "No invoices indexed yet. Please upload PDF invoices first.",
            "sources": [],
        }
    result = engine.query(req.question, req.invoice_data)
    return result


@app.post("/api/rag/explain-anomaly")
async def explain_anomaly(req: AnomalyRequest):
    """Get an LLM explanation for a flagged invoice."""
    explanation = engine.explain_anomaly(req.invoice)
    return {"explanation": explanation}


@app.post("/api/rag/generate-report")
async def generate_report(req: ReportRequest):
    """Auto-generate a financial summary report."""
    report = engine.generate_report(req.invoice_data)
    return {"report": report}


@app.get("/api/rag/status")
async def rag_status():
    """Check how many documents are indexed."""
    return {
        "total_chunks":   vs.count(),
        "indexed_files":  vs.list_invoices(),
        "ready":          vs.count() > 0,
    }


@app.delete("/api/rag/invoice/{invoice_id}")
async def delete_invoice(invoice_id: str):
    """Remove an invoice from the vector store."""
    vs.delete_invoice(invoice_id)
    return {"message": f"Deleted {invoice_id}", "total_docs": vs.count()}


# ─────────────────────────────────────────
# Run with: uvicorn rag_pipeline:app --port 8001
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
