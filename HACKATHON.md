# 📚 Open-Source RAG PDF Chatbot — Hackathon Doc

**Date:** 2026-06-13
**One-liner:** Ask questions over a large private PDF corpus and get answers with
file + page citations in 2–5 seconds — on a 100% free / open-source stack.

---

## The problem
Teams sit on huge piles of PDFs (manuals, reports, contracts) and can't search them by
meaning. Hosted RAG tools are paid, send data to third parties, and hide their sources.

## Our solution
A self-contained RAG service that ingests PDFs, retrieves the most relevant passages,
and generates a grounded answer — **every answer cites the exact file and page**. It
runs end-to-end with zero API keys and zero external services.

## Why it's interesting
- **Fully offline-capable** — no key needed. Falls back to a built-in extractive
  answerer, so the demo always runs.
- **Pluggable generation** — auto-detects Ollama / Anthropic / OpenAI if available, for
  fluent prose; otherwise extractive.
- **Citations on every answer** (file + page) + a live retrieval visualization.
- **Fast** — 2–5 s end-to-end, measured.

---

## How it works (pipeline)

```
PDF → extract (PyMuPDF + OCR fallback) → clean + lang-detect
    → token-aware chunking (800 tok / 150 overlap, page-tagged)
    → embed (all-MiniLM-L6-v2, 384-dim)
    → store (ChromaDB, HNSW, cosine)

Query → embed → ANN retrieve top-K → cross-encoder rerank → keep top-N
      → grounded generation → answer + citations + per-stage timings
```

## Stack (all open-source)
| Stage | Tool |
|-------|------|
| Extraction / OCR | PyMuPDF + Tesseract (optional) |
| Chunking | tiktoken, token-aware, page-tagged |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent HNSW) |
| Reranking | cross-encoder `ms-marco-MiniLM-L-6-v2` |
| Generation | Ollama / Anthropic / OpenAI, or built-in extractive |
| API + UI | FastAPI + single-page chat |

---

## Run it (one command)

```bash
cd prjct_ai
./run.sh          # venv + deps + generates 10×210-page sample corpus + starts server
```

Open **http://localhost:8000** → click **"Ingest data/pdfs/"** → ask a question.

## Demo flow (≈2 min)
1. **Ingest** the sample corpus — watch per-file pages / chunks / OCR / timing.
2. **Ask** a question — answer appears with **citation chips** (file + page) and a
   per-stage latency breakdown.
3. **Inspect** the right panel — top-K retrieved chunks with similarity + rerank scores.

---

## Evaluation
```bash
python evaluate.py
```
Reports latency **p50 / p95**, **Recall@k**, **MRR**, **citation accuracy** against a
gold set, and how many queries land inside the 5 s budget.

## API
`POST /api/ingest` · `POST /api/ingest_dir` · `POST /api/query` ·
`GET /api/stats` · `GET /api/health` · `POST /api/reset`

## Config
All knobs live in `rag/config.py`, overridable via env vars
(`RAG_CHUNK_TOKENS`, `RAG_TOP_K`, `RAG_FINAL_K`, `RAG_USE_RERANKER`, `RAG_EMBED_MODEL`, …).

## Team notes / next steps
- Hybrid (BM25 + dense) retrieval for keyword-heavy queries.
- Streaming token output in the UI.
- Multi-corpus / per-user collections.
