from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Tuple

from . import config

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant for a private PDF knowledge base. "
    "Answer ONLY using the provided context passages. If the answer is not contained "
    "in the context, say you could not find it in the documents. "
    "Cite every claim inline using the format [filename p.PAGE] drawn from the passage "
    "headers. Be concise and factual; never invent sources or page numbers."
)


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        pages = (
            f"p.{meta['page_start']}"
            if meta["page_start"] == meta["page_end"]
            else f"p.{meta['page_start']}-{meta['page_end']}"
        )
        title = f"[{idx}] Source: {meta['filename']} {pages}"
        parts.append(f"{title}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _user_prompt(question: str, context: str) -> str:
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline citations like "
        "[filename p.N]."
    )


def _detect_backend() -> str:
    if config.LLM_BACKEND != "auto":
        return config.LLM_BACKEND
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if _ollama_up():
        return "ollama"
    return "extractive"


def _ollama_up() -> bool:
    try:
        import requests

        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=0.5)
        return True
    except Exception:
        return False


def _gen_anthropic(question: str, context: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    reply = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_prompt(question, context)}],
    )
    return "".join(block.text for block in reply.content if block.type == "text").strip()


def _gen_openai(question: str, context: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    reply = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(question, context)},
        ],
    )
    return reply.choices[0].message.content.strip()


def _gen_ollama(question: str, context: str) -> str:
    import requests

    reply = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(question, context)},
            ],
            "options": {"num_predict": config.LLM_MAX_TOKENS},
        },
        timeout=120,
    )
    reply.raise_for_status()
    return reply.json()["message"]["content"].strip()


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _gen_extractive(question: str, chunks: List[Dict[str, Any]]) -> str:
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}
    ranked: List[Tuple[float, str, str]] = []
    for chunk in chunks:
        meta = chunk["metadata"]
        pages = (
            f"p.{meta['page_start']}"
            if meta["page_start"] == meta["page_end"]
            else f"p.{meta['page_start']}-{meta['page_end']}"
        )
        cite = f"[{meta['filename']} {pages}]"
        for raw in _SENT_RE.split(chunk["text"]):
            sentence = " ".join(raw.split())
            if len(sentence) < 30:
                continue
            tokens = set(re.findall(r"\w+", sentence.lower()))
            matches = len(words & tokens)
            if matches:
                ranked.append((matches + chunk.get("score", 0), sentence, cite))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return (
            "I could not find a relevant answer to that question in the ingested "
            "documents. Try rephrasing or ingesting more PDFs."
        )
    seen, lines = set(), []
    for _, sentence, cite in ranked[:4]:
        key = sentence[:80]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {sentence} {cite}")
    return (
        "Based on the retrieved passages (extractive summary — no LLM key configured):\n"
        + "\n".join(lines)
    )


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    backend = _detect_backend()
    context = _build_context(chunks)
    try:
        if backend == "anthropic":
            answer = _gen_anthropic(question, context)
        elif backend == "openai":
            answer = _gen_openai(question, context)
        elif backend == "ollama":
            answer = _gen_ollama(question, context)
        else:
            answer = _gen_extractive(question, chunks)
    except Exception as err:
        answer = _gen_extractive(question, chunks)
        backend = f"extractive (fallback after {backend} error: {err})"
    return {"answer": answer, "backend": backend}
