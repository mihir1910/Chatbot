# 📚 Open-Source RAG PDF Chatbot

Retrieval-Augmented Generation chatbot that answers questions over a large private
corpus of PDFs (≥10 PDFs, ≥200 pages each) using a **fully free / open-source** stack,
with 2–5 s end-to-end latency and **source citations (file + page)** on every answer.

## Stack (all free / open-source)

| Stage | Tool |
|-------|------|
| PDF text extraction | **PyMuPDF** (native text layer) |
| OCR (scanned pages/images) | **Tesseract** via `pytesseract` (optional, auto-detected) |
| Chunking | token-aware (tiktoken), 800 tokens / 150 overlap, page-tagged |
| Embeddings | **sentence-transformers `all-MiniLM-L6-v2`** (384-dim) |
| Vector DB / ANN | **ChromaDB** (persistent HNSW, cosine) |
| Reranking | **cross-encoder `ms-marco-MiniLM-L-6-v2`** (optional) |
| Generation | Ollama (local) / Anthropic / OpenAI if a key is set — else a built-in **extractive** answerer (no key needed) |
| API + UI | FastAPI + a single-page web chat |

The generation step is the only place a hosted model *can* plug in. With **no API key
and no Ollama**, the system still works end-to-end using an extractive answerer, so the
demo is runnable with zero external dependencies.

## Quick start

```bash
cd prjct_ai
./run.sh                 # creates venv, installs deps, generates sample PDFs, starts server
```

Then open **http://localhost:8000** and click **“Ingest data/pdfs/”**.

### Manual steps (equivalent)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python gen_sample_pdfs.py            # 10 PDFs × 210 pages of queryable prose
uvicorn app:app --port 8000
```

> **OCR:** install the Tesseract binary to enable scanned-page OCR
> (`brew install tesseract` / `apt install tesseract-ocr`). Without it, native text
> extraction still works and OCR is simply skipped.

> **Better answers:** for fluent generated prose instead of extractive bullets, either
> run [Ollama](https://ollama.com) (`ollama run llama3.1:8b`) or set `ANTHROPIC_API_KEY`
> / `OPENAI_API_KEY` in the environment. The backend auto-detects what's available.

## Using the demo

The web UI has three panels that map to the pipeline:

1. **Left — Ingestion:** drag-and-drop PDFs (or ingest `data/pdfs/`). Shows pages,
   chunks, OCR pages, and timing per file. Live corpus stats below.
2. **Center — Chat:** ask questions; answers carry **citation chips** (file + page) and
   a per-stage latency breakdown.
3. **Right — Retrieval visualization:** the top-K retrieved chunks with similarity +
   rerank scores and text previews.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/ingest` | multipart upload of PDF(s) |
| POST | `/api/ingest_dir` | ingest everything in `data/pdfs/` |
| POST | `/api/query` | `{ "question": "..." }` → answer + sources + retrieved + timings |
| GET | `/api/stats` | corpus stats |
| GET | `/api/health` | active models / backend / config |
| POST | `/api/reset` | wipe the vector store |

## Evaluation & monitoring

```bash
python evaluate.py
```

Reports latency **p50 / p95**, **Recall@k**, **MRR**, and **citation accuracy** against
a labelled gold set, plus how many queries land within the 5 s budget.

## Latency tuning

- `all-MiniLM-L6-v2` query embedding is ~10–30 ms on CPU.
- ChromaDB HNSW retrieval over hundreds of thousands of chunks is single-digit ms.
- The cross-encoder reranker adds ~50–150 ms for 8 candidates; disable with
  `RAG_USE_RERANKER=0` if you need to shave latency.
- Generation dominates: extractive ≈ 0 ms; Ollama 8B ≈ 1–3 s; Haiku/4o-mini ≈ 0.5–1.5 s.

All knobs live in [`rag/config.py`](rag/config.py) and are overridable via env vars
(`RAG_CHUNK_TOKENS`, `RAG_TOP_K`, `RAG_FINAL_K`, `RAG_EMBED_MODEL`, …).

## Project layout

```
prjct_ai/
├── app.py                 FastAPI app (API + serves UI)
├── gen_sample_pdfs.py     generate the 10×210-page sample corpus
├── evaluate.py            latency / recall / MRR / citation metrics
├── run.sh                 one-command setup + launch
├── requirements.txt
├── rag/
│   ├── config.py          all tunable settings
│   ├── ingest.py          PDF extract + OCR + clean + lang detect
│   ├── chunking.py        deterministic token-aware, page-tagged chunking
│   ├── embeddings.py      sentence-transformers wrapper
│   ├── vectorstore.py     ChromaDB (HNSW) wrapper
│   ├── rerank.py          cross-encoder reranker
│   ├── generate.py        multi-backend grounded generation + citations
│   └── pipeline.py        ingest + query orchestration with timing
├── static/index.html      chat UI + ingestion + retrieval viz
└── data/{pdfs,chroma}/    corpus + persisted vectors
```
