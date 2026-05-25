"""
ocr_utils.py — Text extraction from PDF / DOCX / image / video files.

Supports: JPEG, PNG, PDF (native + OCR fallback), DOCX, MP4, MOV, AVI.
Temp files are always cleaned up. Invalid files raise clear errors.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import List, Set

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import pdfplumber


logger = logging.getLogger(__name__)


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


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
VIDEO_BLUR_THRESHOLD = 100.0
MIN_VIDEO_TEXT_LENGTH = 15


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


def _frame_blur_score(frame) -> float:
    """Return a Laplacian-variance blur score for a video frame."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def _preprocess_video_frame(frame) -> Image.Image:
    """Preprocess a frame for OCR: grayscale, 2x resize, sharpen, threshold."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(
        gray_frame,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )
    sharpen_kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
    )
    sharpened_frame = cv2.filter2D(resized_frame, -1, sharpen_kernel)
    _, thresholded_frame = cv2.threshold(
        sharpened_frame,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    return Image.fromarray(thresholded_frame)


def _normalize_ocr_text(text: str) -> str:
    """Normalize OCR output so duplicate frame text can be skipped."""
    return " ".join(text.split()).strip().lower()


def _clean_ocr_lines(text: str) -> List[str]:
    """Return cleaned OCR lines that are long enough to be meaningful."""
    lines: List[str] = []
    for raw_line in text.splitlines():
        cleaned_line = " ".join(raw_line.split()).strip()
        if len(cleaned_line) >= MIN_VIDEO_TEXT_LENGTH:
            lines.append(cleaned_line)
    return lines


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


def extract_video_text(video_path: str) -> str:
    """Extract OCR text from video frames sampled every second."""
    _, ext = os.path.splitext(video_path.lower())
    if ext not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video type '{ext}'. Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Corrupted video or unsupported video file.")

    try:
        parts: List[str] = []
        seen_lines: Set[str] = set()
        total_frames_processed = 0
        ocr_text_extracted = 0
        blurry_frames_skipped = 0
        frame_interval_seconds = 1.0
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0

        def _append_unique_lines(frame_text: str) -> None:
            nonlocal ocr_text_extracted
            cleaned_lines = _clean_ocr_lines(frame_text)
            if not cleaned_lines:
                return

            ocr_text_extracted += 1
            for line in cleaned_lines:
                normalized_line = _normalize_ocr_text(line)
                if normalized_line and normalized_line not in seen_lines:
                    seen_lines.add(normalized_line)
                    parts.append(line)

        if fps > 0 and frame_count > 0:
            sample_times = np.arange(0.0, max(duration_seconds, 0.0) + 0.001, frame_interval_seconds)
            for timestamp_seconds in sample_times:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
                success, frame = capture.read()
                if not success:
                    continue
                total_frames_processed += 1
                if _frame_blur_score(frame) < VIDEO_BLUR_THRESHOLD:
                    blurry_frames_skipped += 1
                    continue
                ocr_frame = _preprocess_video_frame(frame)
                frame_text = _ocr_image(ocr_frame, preprocess=False)
                _append_unique_lines(frame_text)
        else:
            max_attempts = 300
            timestamp_seconds = 0.0
            attempts = 0
            while attempts < max_attempts:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
                success, frame = capture.read()
                if not success:
                    if parts:
                        break
                    attempts += 1
                    timestamp_seconds += frame_interval_seconds
                    continue
                total_frames_processed += 1
                if _frame_blur_score(frame) < VIDEO_BLUR_THRESHOLD:
                    blurry_frames_skipped += 1
                    attempts += 1
                    timestamp_seconds += frame_interval_seconds
                    continue
                ocr_frame = _preprocess_video_frame(frame)
                frame_text = _ocr_image(ocr_frame, preprocess=False)
                _append_unique_lines(frame_text)
                attempts += 1
                timestamp_seconds += frame_interval_seconds

        if not parts:
            raise RuntimeError("No meaningful OCR text extracted from video.")

        logger.debug(
            "Video OCR stats for %s: total_frames_processed=%d, ocr_text_extracted=%d, blurry_frames_skipped=%d, unique_lines=%d",
            video_path,
            total_frames_processed,
            ocr_text_extracted,
            blurry_frames_skipped,
            len(parts),
        )

        return "\n\n".join(parts)
    finally:
        capture.release()


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
