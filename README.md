# RAG PDF Chatbot

Ask questions over a pile of PDFs and get answers back with the source file and page
number. Runs on a free, open-source stack and answers in a few seconds. No API key
needed to try it.

## How it works

Ingest side: read the PDF text (with OCR fallback for scanned pages), split it into
~800-token chunks tagged with their page number, embed each chunk, and store the vectors
in ChromaDB.

Query side: embed the question, pull the closest chunks from ChromaDB, rerank them with a
cross-encoder, and generate an answer from the top few. Every answer keeps the file and
page it came from.

Generation uses Ollama, Anthropic, or OpenAI if one is available, and falls back to a
built-in extractive answer otherwise — so it works with nothing installed.

## Stack

- PyMuPDF for text extraction, Tesseract for OCR (optional)
- tiktoken for token-aware chunking
- sentence-transformers `all-MiniLM-L6-v2` for embeddings
- ChromaDB for the vector store
- cross-encoder `ms-marco-MiniLM-L-6-v2` for reranking (optional)
- FastAPI + a single-page chat UI

## Quick start

```bash
cd prjct_ai
./run.sh
```

That sets up a venv, installs deps, generates a sample PDF corpus, and starts the server.
Open http://localhost:8000 and click "Ingest data/pdfs/".

Manual version:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python gen_sample_pdfs.py
uvicorn app:app --port 8000
```

Notes:

- For OCR on scanned pages, install Tesseract (`brew install tesseract` or
  `apt install tesseract-ocr`). Without it, normal text extraction still works.
- For nicer prose instead of extractive answers, run Ollama
  (`ollama run llama3.1:8b`) or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.

## The UI

Three panels: ingestion on the left (drag in PDFs, see pages/chunks/timing), chat in the
middle (answers with citation chips and a latency breakdown), and the retrieved chunks
with their scores on the right.

## API

- `POST /api/ingest` — upload PDFs
- `POST /api/ingest_dir` — ingest everything in `data/pdfs/`
- `POST /api/query` — `{ "question": "..." }`, returns answer + sources + timings
- `GET /api/stats` — corpus stats
- `GET /api/health` — active models and config
- `POST /api/reset` — wipe the vector store

## Evaluation

```bash
python evaluate.py
```

Prints latency p50/p95, Recall@k, MRR, and citation accuracy against a small gold set.

## Config

Everything tunable lives in `rag/config.py` and can be overridden with env vars
(`RAG_CHUNK_TOKENS`, `RAG_TOP_K`, `RAG_FINAL_K`, `RAG_EMBED_MODEL`, `RAG_USE_RERANKER`, …).

## Layout

```
prjct_ai/
├── app.py              FastAPI app + serves the UI
├── gen_sample_pdfs.py  generates the sample corpus
├── evaluate.py         metrics
├── run.sh              setup + launch
├── rag/
│   ├── config.py       settings
│   ├── ingest.py       extract + OCR + clean
│   ├── chunking.py     token-aware, page-tagged chunking
│   ├── embeddings.py   sentence-transformers wrapper
│   ├── vectorstore.py  ChromaDB wrapper
│   ├── rerank.py       cross-encoder reranker
│   ├── generate.py     answer generation + citations
│   └── pipeline.py     ingest + query orchestration
├── static/index.html   chat UI
└── data/               PDFs + persisted vectors
```
