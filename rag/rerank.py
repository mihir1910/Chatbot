from . import config

reranker = None
load_failed = False


def get_reranker():
    global reranker, load_failed
    if reranker is None and not load_failed:
        try:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(config.RERANK_MODEL)
        except Exception:
            load_failed = True
    return reranker


def rerank(query, candidates, top_n):
    if not config.USE_RERANKER or not candidates:
        return candidates[:top_n]

    model = get_reranker()
    if model is None:
        return candidates[:top_n]

    pairs = [(query, item["text"]) for item in candidates]
    scores = model.predict(pairs)
    for item, score in zip(candidates, scores):
        item["rerank_score"] = round(float(score), 4)
    ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_n]
