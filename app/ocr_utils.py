"""
ocr_utils.py — Text extraction from PDF / DOCX / image files.

Supports: JPEG, PNG, PDF (native + OCR fallback), DOCX.
Temp files are always cleaned up. Invalid files raise clear errors.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import List

import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import pdfplumber

TESSERACT_PATHS = [
    r"E:\GENAI\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
]

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break


def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


try:
    import docx2txt
    _DOCX2TXT = True
except ImportError:
    _DOCX2TXT = False

try:
    from pdf2image import convert_from_bytes
    _PDF2IMAGE = True
except ImportError:
    _PDF2IMAGE = False


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_image(img: Image.Image) -> Image.Image:
    """Improve OCR quality: greyscale → sharpen → contrast boost."""
    img = img.convert("L")                          # greyscale
    img = img.filter(ImageFilter.SHARPEN)           # sharpen
    img = ImageEnhance.Contrast(img).enhance(2.0)   # boost contrast
    return img


# ── Core OCR helpers ──────────────────────────────────────────────────────────

def _ocr_image(img: Image.Image, preprocess: bool = True) -> str:
    """Run Tesseract on a PIL Image; optionally pre-process first."""
    if not check_tesseract():
        raise RuntimeError(
            "Tesseract OCR is not installed or configured correctly."
        )
    if preprocess:
        img = _preprocess_image(img)
    return pytesseract.image_to_string(img)


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract PDF text; fall back to per-page OCR for image-only pages."""
    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
            elif _PDF2IMAGE:
                imgs: List[Image.Image] = convert_from_bytes(
                    file_bytes, first_page=i + 1, last_page=i + 1, dpi=200
                )
                for img in imgs:
                    parts.append(_ocr_image(img))
    return "\n\n".join(parts)


def _extract_docx(file_bytes: bytes) -> str:
    """Extract DOCX text via docx2txt with guaranteed temp-file cleanup."""
    if not _DOCX2TXT:
        raise ImportError("docx2txt is not installed.")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        return docx2txt.process(tmp_path) or ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Public API ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".docx"}


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from an uploaded file.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename; extension determines the extraction path.

    Returns
    -------
    str
        Extracted plain text.

    Raises
    ------
    ValueError
        For unsupported file types or zero-byte files.
    RuntimeError
        If extraction fails internally.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if ext in {".png", ".jpg", ".jpeg"}:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()                          # detect corrupt images early
            img = Image.open(io.BytesIO(file_bytes))  # re-open after verify
            return _ocr_image(img)

        elif ext == ".pdf":
            return _extract_pdf(file_bytes)

        elif ext == ".docx":
            return _extract_docx(file_bytes)

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Text extraction error ({ext}): {exc}") from exc

    return ""
