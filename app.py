from __future__ import annotations

import shutil
import traceback
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import config
from rag import pipeline
from rag.vectorstore import get_store
from rag.chunking import token_mode
from rag.ingest import OCR_AVAILABLE
from rag.generate import detect_backend

app = FastAPI(title="Open-Source RAG PDF Chatbot")


@app.on_event("startup")
def warmup():
    try:
        from rag import embeddings, rerank

        embeddings.embed_query("warmup")
        rerank.rerank("warmup", [{"text": "warmup passage", "score": 0.0}], 1)
        get_store()
        print("[warmup] models + vector store ready")
    except Exception as err:
        print(f"[warmup] skipped: {err}")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    final_k: Optional[int] = None


@app.get("/")
def index():
    return FileResponse(str(config.STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "embed_model": config.EMBED_MODEL,
        "reranker": config.RERANK_MODEL if config.USE_RERANKER else "disabled",
        "llm_backend": detect_backend(),
        "ocr_available": OCR_AVAILABLE,
        "token_mode": token_mode(),
        "chunk_tokens": config.CHUNK_TOKENS,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


@app.get("/api/stats")
def stats():
    return get_store().stats()


@app.post("/api/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    done = []
    for doc in files:
        if not doc.filename.lower().endswith(".pdf"):
            done.append({"filename": doc.filename, "status": "skipped (not a PDF)"})
            continue
        path = config.PDF_DIR / doc.filename
        with open(path, "wb") as out:
            shutil.copyfileobj(doc.file, out)
        try:
            done.append(pipeline.ingest_pdf(str(path), doc.filename))
        except Exception as err:
            traceback.print_exc()
            done.append({"filename": doc.filename, "status": f"error: {err}"})
    return {"results": done, "stats": get_store().stats()}


@app.post("/api/ingest_dir")
def ingest_dir():
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(404, f"No PDFs found in {config.PDF_DIR}")
    done = []
    for pdf in pdfs:
        try:
            done.append(pipeline.ingest_pdf(str(pdf), pdf.name))
        except Exception as err:
            traceback.print_exc()
            done.append({"filename": pdf.name, "status": f"error: {err}"})
    return {"results": done, "stats": get_store().stats()}


@app.post("/api/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    try:
        return pipeline.answer_query(req.question, req.top_k, req.final_k)
    except Exception as err:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(err)})


@app.post("/api/reset")
def reset():
    get_store().reset()
    return {"status": "reset", "stats": get_store().stats()}


app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
