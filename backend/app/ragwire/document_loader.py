"""Document ingestion: extracts text from PDFs and images fully locally.

PyMuPDF (fitz) is used for fast native text extraction. Pages with little or
no extractable text (scanned pages) are rasterized and passed through local
OCR (pytesseract). Plain images go straight to OCR. pdfplumber is used as a
secondary extractor for table-heavy lab reports where PyMuPDF's layout can
mangle columns.
"""
from __future__ import annotations

import io
import logging

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

if settings.tesseract_cmd:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

MIN_CHARS_BEFORE_OCR = 20


def _ocr_image(image: Image.Image) -> str:
    try:
        import pytesseract

        return pytesseract.image_to_string(image)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local OCR failed (%s); returning empty text for this page.", exc)
        return ""


def extract_text_from_image_bytes(content: bytes) -> str:
    image = Image.open(io.BytesIO(content)).convert("RGB")
    return _ocr_image(image)


def extract_text_from_pdf_bytes(content: bytes) -> str:
    import fitz  # PyMuPDF

    text_parts: list[str] = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text().strip()
            if len(page_text) < MIN_CHARS_BEFORE_OCR:
                pixmap = page.get_pixmap(dpi=200)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                ocr_text = _ocr_image(image)
                page_text = ocr_text.strip() or page_text
            text_parts.append(page_text)

    combined = "\n\n".join(text_parts).strip()
    if combined:
        return combined

    # Fall back to pdfplumber for documents where PyMuPDF extracted nothing
    # useful (e.g. unusual encodings) but pages weren't image-only either.
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber fallback failed (%s).", exc)
        return combined


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf_bytes(content)
    if lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        return extract_text_from_image_bytes(content)
    if lower.endswith((".txt", ".md")):
        return content.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported document type: {filename}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Simple sliding-window chunker over whitespace-normalized paragraphs."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = end - overlap
    return chunks
