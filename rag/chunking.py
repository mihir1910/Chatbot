from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import config
from .ingest import DocRecord

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _encode(text: str) -> List[int]:
        return _ENC.encode(text)

    def _decode(tokens: List[int]) -> str:
        return _ENC.decode(tokens)

    _TOKEN_MODE = "tiktoken"
except Exception:
    def _encode(text: str):
        return text.split()

    def _decode(tokens):
        return " ".join(tokens)

    _TOKEN_MODE = "words"


@dataclass
class Chunk:
    chunk_id: str
    pdf_id: str
    filename: str
    text: str
    page_start: int
    page_end: int
    index: int


def chunk_document(doc: DocRecord) -> List[Chunk]:
    toks: List = []
    pages: List[int] = []
    for page in doc.pages:
        encoded = _encode(page.text + "\n\n")
        toks.extend(encoded)
        pages.extend([page.page] * len(encoded))

    result: List[Chunk] = []
    if not toks:
        return result

    stride = max(1, config.CHUNK_TOKENS - config.CHUNK_OVERLAP)
    count = 0
    cursor = 0
    total = len(toks)
    while cursor < total:
        stop = min(cursor + config.CHUNK_TOKENS, total)
        window = toks[cursor:stop]
        body = _decode(window).strip()
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


def token_mode() -> str:
    return _TOKEN_MODE
