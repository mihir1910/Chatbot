import chromadb
from chromadb.config import Settings

from . import config
from .chunking import Chunk


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": config.HNSW_SPACE},
        )

    def add_chunks(self, chunks, embeddings):
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "pdf_id": c.pdf_id,
                    "filename": c.filename,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "index": c.index,
                }
                for c in chunks
            ],
        )

    def has_pdf(self, pdf_id):
        try:
            found = self.collection.get(where={"pdf_id": pdf_id}, limit=1)
            return len(found.get("ids", [])) > 0
        except Exception:
            return False

    def query(self, query_embedding, top_k):
        found = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        if not found["ids"] or not found["ids"][0]:
            return results
        for doc, meta, dist in zip(
            found["documents"][0], found["metadatas"][0], found["distances"][0]
        ):
            results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": round(1.0 - float(dist), 4),
                }
            )
        return results

    def stats(self):
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        seen = {}
        try:
            found = self.collection.get(include=["metadatas"])
            for meta in found.get("metadatas", []):
                name = meta.get("filename", "?")
                entry = seen.setdefault(
                    name, {"filename": name, "chunks": 0, "max_page": 0}
                )
                entry["chunks"] += 1
                entry["max_page"] = max(entry["max_page"], meta.get("page_end", 0))
        except Exception:
            pass
        return {
            "total_chunks": count,
            "num_pdfs": len(seen),
            "pdfs": sorted(seen.values(), key=lambda x: x["filename"]),
        }

    def reset(self):
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": config.HNSW_SPACE},
        )


store = None


def get_store():
    global store
    if store is None:
        store = VectorStore()
    return store
