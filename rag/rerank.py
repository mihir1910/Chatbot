from __future__ import annotations

from typing import List, Dict, Any

from . import config

_reranker = None
_load_failed = False


def _get_reranker():
    global _reranker, _load_failed
    if _reranker is None and not _load_failed:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(config.RERANK_MODEL)
        except Exception:
            _load_failed = True
    return _reranker


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not config.USE_RERANKER or not candidates:
        return candidates[:top_n]

    model = _get_reranker()
    if model is None:
        return candidates[:top_n]

    pairs = [(query, item["text"]) for item in candidates]
    scores = model.predict(pairs)
    for item, score in zip(candidates, scores):
        item["rerank_score"] = round(float(score), 4)
    ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_n]
