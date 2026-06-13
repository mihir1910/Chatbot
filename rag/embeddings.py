from . import config

model = None


def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.EMBED_MODEL)
    return model


def embed_texts(texts):
    encoder = get_model()
    vectors = encoder.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text):
    return embed_texts([text])[0]


def dimension():
    return get_model().get_sentence_embedding_dimension()
