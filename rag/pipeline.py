from __future__ import annotations

import time
from typing import Dict, Any, List

from . import config, embeddings, generate, rerank
from .ingest import extract_pdf
from .chunking import chunk_document
from .vectorstore import get_store


def ingest_pdf(path: str, filename: str | None = None, skip_if_present: bool = True) -> Dict[str, Any]:
    start = time.time()
    doc = extract_pdf(path, filename)
    store = get_store()

    if skip_if_present and store.has_pdf(doc.pdf_id):
        return {
            "filename": doc.filename,
            "pdf_id": doc.pdf_id,
            "status": "skipped (already ingested)",
            "pages": doc.n_pages,
            "chunks": 0,
            "ocr_pages": doc.ocr_pages,
            "seconds": round(time.time() - start, 2),
        }

    extracted = time.time()
    chunks = chunk_document(doc)
    if chunks:
        vecs = embeddings.embed_texts([c.text for c in chunks])
        store.add_chunks(chunks, vecs)
    done = time.time()

    return {
        "filename": doc.filename,
        "pdf_id": doc.pdf_id,
        "status": "ingested",
        "pages": doc.n_pages,
        "ocr_pages": doc.ocr_pages,
        "chunks": len(chunks),
        "extract_seconds": round(extracted - start, 2),
        "embed_seconds": round(done - extracted, 2),
        "seconds": round(done - start, 2),
    }


def answer_query(question: str, top_k: int | None = None, final_k: int | None = None) -> Dict[str, Any]:
    top_k = top_k or config.TOP_K
    final_k = final_k or config.FINAL_K
    timings: Dict[str, float] = {}

    start = time.time()
    q_vec = embeddings.embed_query(question)
    timings["embed_ms"] = round((time.time() - start) * 1000, 1)

    after_embed = time.time()
    candidates = get_store().query(q_vec, top_k)
    timings["retrieve_ms"] = round((time.time() - after_embed) * 1000, 1)

    if not candidates:
        return {
            "question": question,
            "answer": "No documents have been ingested yet, or nothing matched. "
            "Please ingest PDFs first.",
            "backend": "none",
            "sources": [],
            "retrieved": [],
            "timings": timings,
            "total_ms": round((time.time() - start) * 1000, 1),
        }

    after_retrieve = time.time()
    top = rerank.rerank(question, candidates, final_k)
    timings["rerank_ms"] = round((time.time() - after_retrieve) * 1000, 1)

    after_rerank = time.time()
    gen = generate.generate_answer(question, top)
    timings["generate_ms"] = round((time.time() - after_rerank) * 1000, 1)

    sources = _dedup_sources(top)
    total_ms = round((time.time() - start) * 1000, 1)

    return {
        "question": question,
        "answer": gen["answer"],
        "backend": gen["backend"],
        "sources": sources,
        "retrieved": [_view(c) for c in top],
        "candidates_considered": len(candidates),
        "timings": timings,
        "total_ms": total_ms,
    }


def _view(c: Dict[str, Any]) -> Dict[str, Any]:
    meta = c["metadata"]
    return {
        "filename": meta["filename"],
        "page_start": meta["page_start"],
        "page_end": meta["page_end"],
        "score": c.get("score"),
        "rerank_score": c.get("rerank_score"),
        "preview": c["text"][:400] + ("…" if len(c["text"]) > 400 else ""),
    }


def _dedup_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for c in chunks:
        meta = c["metadata"]
        key = (meta["filename"], meta["page_start"], meta["page_end"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "filename": meta["filename"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
            }
        )
    return out
