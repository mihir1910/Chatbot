import os
import re

from . import config

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant for a private PDF knowledge base. "
    "Answer ONLY using the provided context passages. If the answer is not contained "
    "in the context, say you could not find it in the documents. "
    "Cite every claim inline using the format [filename p.PAGE] drawn from the passage "
    "headers. Be concise and factual; never invent sources or page numbers."
)


def build_context(chunks):
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


def user_prompt(question, context):
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline citations like "
        "[filename p.N]."
    )


def detect_backend():
    if config.LLM_BACKEND != "auto":
        return config.LLM_BACKEND
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if ollama_up():
        return "ollama"
    return "extractive"


def ollama_up():
    try:
        import requests

        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=0.5)
        return True
    except Exception:
        return False


def gen_anthropic(question, context):
    import anthropic

    client = anthropic.Anthropic()
    reply = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt(question, context)}],
    )
    return "".join(block.text for block in reply.content if block.type == "text").strip()


def gen_openai(question, context):
    from openai import OpenAI

    client = OpenAI()
    reply = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(question, context)},
        ],
    )
    return reply.choices[0].message.content.strip()


def gen_ollama(question, context):
    import requests

    reply = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, context)},
            ],
            "options": {"num_predict": config.LLM_MAX_TOKENS},
        },
        timeout=120,
    )
    reply.raise_for_status()
    return reply.json()["message"]["content"].strip()


sent_re = re.compile(r"(?<=[.!?])\s+")


def gen_extractive(question, chunks):
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}
    ranked = []
    for chunk in chunks:
        meta = chunk["metadata"]
        pages = (
            f"p.{meta['page_start']}"
            if meta["page_start"] == meta["page_end"]
            else f"p.{meta['page_start']}-{meta['page_end']}"
        )
        cite = f"[{meta['filename']} {pages}]"
        for raw in sent_re.split(chunk["text"]):
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


def generate_answer(question, chunks):
    backend = detect_backend()
    context = build_context(chunks)
    try:
        if backend == "anthropic":
            answer = gen_anthropic(question, context)
        elif backend == "openai":
            answer = gen_openai(question, context)
        elif backend == "ollama":
            answer = gen_ollama(question, context)
        else:
            answer = gen_extractive(question, chunks)
    except Exception as err:
        answer = gen_extractive(question, chunks)
        backend = f"extractive (fallback after {backend} error: {err})"
    return {"answer": answer, "backend": backend}
