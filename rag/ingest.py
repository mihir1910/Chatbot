from __future__ import annotations

import re
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import List

import fitz

from . import config

try:
    import pytesseract
    from PIL import Image
    import io

    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

try:
    from langdetect import detect as _detect_lang
except Exception:
    _detect_lang = None


@dataclass
class PageRecord:
    page: int
    text: str
    source: str
    lang: str = "unknown"


@dataclass
class DocRecord:
    pdf_id: str
    filename: str
    pages: List[PageRecord] = field(default_factory=list)
    n_pages: int = 0
    ocr_pages: int = 0


_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = _HYPHEN_RE.sub(r"\1\2", text)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def _detect_headers_footers(pages: List[str]) -> set:
    counts = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if not lines:
            continue
        for ln in lines[:2] + lines[-2:]:
            if len(ln) < 80:
                counts[ln] += 1
    threshold = max(3, int(0.4 * len(pages)))
    return {ln for ln, c in counts.items() if c >= threshold}


def _strip_chrome(text: str, chrome: set) -> str:
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if s in chrome:
            continue
        if re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", s):
            continue
        kept.append(ln)
    return "\n".join(kept)


def _ocr_page(page: "fitz.Page") -> str:
    if not OCR_AVAILABLE:
        return ""
    pix = page.get_pixmap(dpi=config.OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _safe_lang(text: str) -> str:
    if not _detect_lang or len(text) < 20:
        return "unknown"
    try:
        return _detect_lang(text)
    except Exception:
        return "unknown"


def extract_pdf(path: str, filename: str | None = None) -> DocRecord:
    filename = filename or path.split("/")[-1]
    pdf_id = hashlib.sha1(filename.encode()).hexdigest()[:12]

    doc = fitz.open(path)
    raw_pages: List[tuple[str, str]] = []

    for page in doc:
        native = page.get_text("text") or ""
        if len(native.strip()) >= config.OCR_TEXT_THRESHOLD:
            raw_pages.append((native, "native"))
        else:
            ocr = _ocr_page(page)
            if len(ocr.strip()) > len(native.strip()):
                raw_pages.append((ocr, "ocr"))
            else:
                raw_pages.append((native, "native"))
    doc.close()

    chrome = _detect_headers_footers([t for t, _ in raw_pages])

    rec = DocRecord(pdf_id=pdf_id, filename=filename, n_pages=len(raw_pages))
    for i, (text, source) in enumerate(raw_pages, start=1):
        cleaned = clean_text(_strip_chrome(text, chrome))
        if not cleaned:
            continue
        rec.pages.append(
            PageRecord(page=i, text=cleaned, source=source, lang=_safe_lang(cleaned))
        )
        if source == "ocr":
            rec.ocr_pages += 1
    return rec
