from dataclasses import dataclass

from . import config
from .ingest import DocRecord

try:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    def encode(text):
        return enc.encode(text)

    def decode(tokens):
        return enc.decode(tokens)

    token_mode_value = "tiktoken"
except Exception:
    def encode(text):
        return text.split()

    def decode(tokens):
        return " ".join(tokens)

    token_mode_value = "words"


@dataclass
class Chunk:
    chunk_id: str
    pdf_id: str
    filename: str
    text: str
    page_start: int
    page_end: int
    index: int


def chunk_document(doc):
    toks = []
    pages = []
    for page in doc.pages:
        encoded = encode(page.text + "\n\n")
        toks.extend(encoded)
        pages.extend([page.page] * len(encoded))

    result = []
    if not toks:
        return result

    stride = max(1, config.CHUNK_TOKENS - config.CHUNK_OVERLAP)
    count = 0
    cursor = 0
    total = len(toks)
    while cursor < total:
        stop = min(cursor + config.CHUNK_TOKENS, total)
        window = toks[cursor:stop]
        body = decode(window).strip()
        if len(body) >= config.MIN_CHUNK_CHARS:
            spanned = pages[cursor:stop]
            result.append(
                Chunk(
                    chunk_id=f"{doc.pdf_id}:{count}",
                    pdf_id=doc.pdf_id,
                    filename=doc.filename,
                    text=body,
                    page_start=min(spanned),
                    page_end=max(spanned),
                    index=count,
                )
            )
            count += 1
        if stop == total:
            break
        cursor += stride
    return result


def token_mode():
    return token_mode_value
