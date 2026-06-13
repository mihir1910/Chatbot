from __future__ import annotations

from typing import List

from . import config

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    encoder = get_model()
    vectors = encoder.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


def dimension() -> int:
    return get_model().get_sentence_embedding_dimension()
