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
    from langdetect import detect as detect_lang
except Exception:
    detect_lang = None


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


ws_re = re.compile(r"[ \t]+")
multi_nl_re = re.compile(r"\n{3,}")
hyphen_re = re.compile(r"(\w)-\n(\w)")


def clean_text(text):
    text = text.replace("\x00", " ")
    text = hyphen_re.sub(r"\1\2", text)
    text = ws_re.sub(" ", text)
    text = multi_nl_re.sub("\n\n", text)
    return text.strip()


def detect_headers_footers(pages):
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


def strip_chrome(text, chrome):
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if s in chrome:
            continue
        if re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", s):
            continue
        kept.append(ln)
    return "\n".join(kept)


def ocr_page(page):
    if not OCR_AVAILABLE:
        return ""
    pix = page.get_pixmap(dpi=config.OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def safe_lang(text):
    if not detect_lang or len(text) < 20:
        return "unknown"
    try:
        return detect_lang(text)
    except Exception:
        return "unknown"


def extract_pdf(path, filename=None):
    filename = filename or path.split("/")[-1]
    pdf_id = hashlib.sha1(filename.encode()).hexdigest()[:12]

    doc = fitz.open(path)
    raw_pages = []

    for page in doc:
        native = page.get_text("text") or ""
        if len(native.strip()) >= config.OCR_TEXT_THRESHOLD:
            raw_pages.append((native, "native"))
        else:
            ocr = ocr_page(page)
            if len(ocr.strip()) > len(native.strip()):
                raw_pages.append((ocr, "ocr"))
            else:
                raw_pages.append((native, "native"))
    doc.close()

    chrome = detect_headers_footers([t for t, _ in raw_pages])

    rec = DocRecord(pdf_id=pdf_id, filename=filename, n_pages=len(raw_pages))
    for i, (text, source) in enumerate(raw_pages, start=1):
        cleaned = clean_text(strip_chrome(text, chrome))
        if not cleaned:
            continue
        rec.pages.append(
            PageRecord(page=i, text=cleaned, source=source, lang=safe_lang(cleaned))
        )
        if source == "ocr":
            rec.ocr_pages += 1
    return rec
