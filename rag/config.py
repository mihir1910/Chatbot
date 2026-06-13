import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
STATIC_DIR = BASE_DIR / "static"

for path in (DATA_DIR, PDF_DIR, CHROMA_DIR):
    path.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH", "64"))

CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
MIN_CHUNK_CHARS = 60

COLLECTION_NAME = os.getenv("RAG_COLLECTION", "pdf_chunks")
TOP_K = int(os.getenv("RAG_TOP_K", "8"))
FINAL_K = int(os.getenv("RAG_FINAL_K", "4"))
HNSW_SPACE = "cosine"

USE_RERANKER = os.getenv("RAG_USE_RERANKER", "1") == "1"
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

OCR_TEXT_THRESHOLD = int(os.getenv("RAG_OCR_THRESHOLD", "100"))
OCR_DPI = int(os.getenv("RAG_OCR_DPI", "200"))

LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "auto")
ANTHROPIC_MODEL = os.getenv("RAG_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MAX_TOKENS = int(os.getenv("RAG_LLM_MAX_TOKENS", "700"))
