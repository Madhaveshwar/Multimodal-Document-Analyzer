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
from typing import List, Optional, Set, Tuple

import numpy as np
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pytesseract import Output

from config import get_settings
from logging_utils import get_logger, log_ocr
from tracing_utils import traceable

try:
    import cv2
except Exception:
    cv2 = None

try:
    import streamlit as st
except Exception:
    st = None

logger = get_logger(__name__)
SETTINGS = get_settings()
TESSERACT_CMD = SETTINGS.tesseract_cmd
MIN_IMAGE_TEXT_LENGTH = 10
MIN_VIDEO_TEXT_LENGTH = 15
VIDEO_BLUR_THRESHOLD = 75.0
VIDEO_EMPTY_STD_THRESHOLD = 8.0
VIDEO_BASE_INTERVAL_SECONDS = 3.0
VIDEO_MAX_SAMPLES = 120
OCR_CONFIDENCE_RETRY_THRESHOLD = 45.0
OCR_LOW_TEXT_RETRY_LENGTH = 24

_TESSERACT_READY = False
_TESSERACT_ERROR_MESSAGE: Optional[str] = None


def _notify_streamlit_error(message: str) -> None:
    if st is not None:
        try:
            st.error(message)
        except Exception:
            pass


def _set_tesseract_error(message: str, exc: Optional[Exception] = None) -> None:
    global _TESSERACT_READY, _TESSERACT_ERROR_MESSAGE
    _TESSERACT_READY = False
    _TESSERACT_ERROR_MESSAGE = message
    logger.error(message)
    log_ocr(message, error=True)
    if exc is not None:
        logger.debug("Tesseract validation failure details: %s", exc)


def _configure_tesseract() -> None:
    global _TESSERACT_READY, _TESSERACT_ERROR_MESSAGE

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    if not os.path.isfile(TESSERACT_CMD):
        _set_tesseract_error(
            f"Tesseract OCR executable was not found at '{TESSERACT_CMD}'. "
            "Install Tesseract there or update the configured path."
        )
        return

    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:
        _set_tesseract_error(
            f"Tesseract OCR is installed at '{TESSERACT_CMD}' but pytesseract could not access it. "
            "Verify the executable is readable and not blocked by Windows security settings.",
            exc,
        )
        return

    _TESSERACT_READY = True
    _TESSERACT_ERROR_MESSAGE = None
    logger.debug("Tesseract configured successfully: %s", version)


def _ensure_tesseract_ready() -> None:
    if not _TESSERACT_READY:
        message = _TESSERACT_ERROR_MESSAGE or (
            f"Tesseract OCR is not installed or configured correctly at '{TESSERACT_CMD}'."
        )
        _notify_streamlit_error(message)
        raise RuntimeError(message)


_configure_tesseract()


def check_tesseract():
    return _TESSERACT_READY


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


def _resize_for_ocr(img: Image.Image, aggressive: bool = False) -> Image.Image:
    """Upscale smaller images before OCR so Tesseract has more usable detail."""
    width, height = img.size
    min_dimension = min(width, height)
    target_min_dimension = 1400 if aggressive else 1100
    if min_dimension >= target_min_dimension:
        return img

    scale = target_min_dimension / max(min_dimension, 1)
    if aggressive:
        scale *= 1.25

    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return img.resize(new_size, Image.LANCZOS)


def _enhance_array_for_ocr(gray_array: np.ndarray, aggressive: bool = False) -> np.ndarray:
    if gray_array.dtype != np.uint8:
        gray_array = gray_array.astype(np.uint8)

    if cv2 is None:
        fallback_image = Image.fromarray(gray_array)
        fallback_image = ImageOps.autocontrast(fallback_image)
        fallback_image = ImageEnhance.Sharpness(fallback_image).enhance(2.2 if aggressive else 1.7)
        if aggressive:
            fallback_image = fallback_image.filter(ImageFilter.MedianFilter(size=3))
        return np.array(fallback_image)

    denoise_strength = 14 if aggressive else 9
    contrast = cv2.fastNlMeansDenoising(gray_array, None, denoise_strength, 7, 21)

    if aggressive:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast = clahe.apply(contrast)
    else:
        contrast = cv2.equalizeHist(contrast)

    sharpen_kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
    )
    sharpened = cv2.filter2D(contrast, -1, sharpen_kernel)

    block_size = 35 if aggressive else 31
    threshold_c = 12 if aggressive else 11
    thresholded = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        threshold_c,
    )
    return thresholded


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_image(img: Image.Image, aggressive: bool = False) -> Image.Image:
    """Improve OCR quality with resize, denoise, contrast, sharpen, and thresholding."""
    prepared = img.convert("RGB")
    prepared = _resize_for_ocr(prepared, aggressive=aggressive)
    prepared = ImageOps.grayscale(prepared)
    prepared = ImageEnhance.Contrast(prepared).enhance(2.1 if aggressive else 1.8)
    gray_array = np.array(prepared)
    enhanced_array = _enhance_array_for_ocr(gray_array, aggressive=aggressive)
    return Image.fromarray(enhanced_array)


def _preprocess_video_frame(frame: np.ndarray, aggressive: bool = False) -> Image.Image:
    """Preprocess a video frame for OCR with resize, denoise, sharpen, and threshold."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for video OCR preprocessing but is not installed in this environment."
        )
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)
    return _preprocess_image(frame_image, aggressive=aggressive)


def _frame_blur_score(frame: np.ndarray) -> float:
    """Return a Laplacian-variance blur score for a video frame."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for video OCR preprocessing but is not installed in this environment."
        )
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def _frame_empty_score(frame: np.ndarray) -> float:
    """Return a low-score indicator for nearly empty or uniform frames."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for video OCR preprocessing but is not installed in this environment."
        )
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray_frame))


# ── Core OCR helpers ──────────────────────────────────────────────────────────

def _run_tesseract(img: Image.Image, psm: int = 6) -> Tuple[str, float]:
    """Run Tesseract and return merged text plus a mean confidence score."""
    _ensure_tesseract_ready()

    config = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
    except Exception as exc:
        raise RuntimeError(f"Tesseract OCR failed: {exc}") from exc

    words: List[str] = []
    confidences: List[float] = []

    for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
        text = str(raw_text).strip()
        if not text:
            continue

        words.append(text)
        try:
            confidence = float(raw_conf)
        except Exception:
            continue
        if confidence >= 0:
            confidences.append(confidence)

    merged_text = " ".join(words).strip()
    if not merged_text:
        try:
            merged_text = pytesseract.image_to_string(img, config=config).strip()
        except Exception as exc:
            raise RuntimeError(f"Tesseract OCR failed: {exc}") from exc

    average_confidence = float(sum(confidences) / len(confidences)) if confidences else -1.0
    return merged_text, average_confidence


def _ocr_image(
    img: Image.Image,
    preprocess: bool = True,
    aggressive: bool = False,
    psm: int = 6,
) -> Tuple[str, float]:
    """Run Tesseract on a PIL Image; optionally pre-process first."""
    if preprocess:
        img = _preprocess_image(img, aggressive=aggressive)
    return _run_tesseract(img, psm=psm)


def _is_meaningful_text(text: str, minimum_length: int) -> bool:
    return len(" ".join(text.split()).strip()) >= minimum_length


def _normalize_ocr_text(text: str) -> str:
    """Normalize OCR output so duplicate frame text can be skipped."""
    return " ".join(text.split()).strip().lower()


def _clean_ocr_lines(text: str, minimum_length: int) -> List[str]:
    """Return cleaned OCR lines that are long enough to be meaningful."""
    lines: List[str] = []
    for raw_line in text.splitlines():
        cleaned_line = " ".join(raw_line.split()).strip()
        if len(cleaned_line) >= minimum_length:
            lines.append(cleaned_line)
    return lines


def _should_retry_ocr(text: str, confidence: float) -> bool:
    normalized_text_length = len(_normalize_ocr_text(text))
    if normalized_text_length < OCR_LOW_TEXT_RETRY_LENGTH:
        return True
    return confidence >= 0 and confidence < OCR_CONFIDENCE_RETRY_THRESHOLD


def _prefer_ocr_candidate(
    current_text: str,
    current_confidence: float,
    candidate_text: str,
    candidate_confidence: float,
) -> bool:
    current_length = len(_normalize_ocr_text(current_text))
    candidate_length = len(_normalize_ocr_text(candidate_text))

    if candidate_length == 0:
        return False
    if current_length == 0:
        return True
    if candidate_confidence >= 0 and current_confidence >= 0:
        if candidate_confidence >= current_confidence + 5.0:
            return True
    if candidate_length > current_length * 1.2:
        return True
    if current_confidence < 0 <= candidate_confidence:
        return True
    return False


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract PDF text; fall back to per-page OCR for image-only pages."""
    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
                continue

            if not _PDF2IMAGE:
                logger.debug("PDF OCR fallback unavailable for page %d because pdf2image is missing", page_index)
                continue

            try:
                imgs: List[Image.Image] = convert_from_bytes(
                    file_bytes,
                    first_page=page_index,
                    last_page=page_index,
                    dpi=240,
                )
            except Exception as exc:
                logger.warning("PDF rasterization failed on page %d: %s", page_index, exc)
                continue

            for img in imgs:
                try:
                    ocr_text, _ = _ocr_image(img, preprocess=True, aggressive=True, psm=6)
                    if _is_meaningful_text(ocr_text, MIN_IMAGE_TEXT_LENGTH):
                        parts.append(ocr_text)
                except Exception as exc:
                    logger.warning("PDF OCR failed on page %d: %s", page_index, exc)

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


@traceable(name="Video OCR", run_type="tool")
def extract_video_text(video_path: str) -> str:
    """Extract OCR text from video frames sampled every few seconds."""
    _ensure_tesseract_ready()

    if cv2 is None:
        message = (
            "OpenCV is not installed, so video OCR cannot run in this environment. "
            "Install opencv-python to enable MP4, MOV, and AVI extraction."
        )
        _notify_streamlit_error(message)
        raise RuntimeError(message)

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
        seen_blocks: Set[str] = set()
        seen_lines: Set[str] = set()
        total_frames_processed = 0
        sampled_frames = 0
        frames_kept = 0
        blurry_frames_skipped = 0
        empty_frames_skipped = 0
        ocr_failures = 0
        retry_count = 0
        frame_interval_seconds = VIDEO_BASE_INTERVAL_SECONDS
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0

        def _append_unique_lines(frame_text: str) -> bool:
            nonlocal frames_kept
            cleaned_lines = _clean_ocr_lines(frame_text, MIN_VIDEO_TEXT_LENGTH)
            if not cleaned_lines:
                return False

            candidate_lines: List[str] = []
            for line in cleaned_lines:
                normalized_line = _normalize_ocr_text(line)
                if normalized_line and normalized_line not in seen_lines:
                    seen_lines.add(normalized_line)
                    candidate_lines.append(line)

            if not candidate_lines:
                return False

            candidate_block = "\n".join(candidate_lines)
            normalized_block = _normalize_ocr_text(candidate_block)
            if not normalized_block or normalized_block in seen_blocks:
                return False

            seen_blocks.add(normalized_block)
            parts.append(candidate_block)
            frames_kept += 1
            return True

        def _process_frame(frame: np.ndarray, frame_index: int, timestamp_seconds: float) -> None:
            nonlocal sampled_frames, blurry_frames_skipped, empty_frames_skipped, ocr_failures, retry_count
            sampled_frames += 1

            blur_score = _frame_blur_score(frame)
            if blur_score < VIDEO_BLUR_THRESHOLD:
                blurry_frames_skipped += 1
                logger.debug(
                    "Video OCR skipped blurry frame %d at %.2fs (blur_score=%.2f)",
                    frame_index,
                    timestamp_seconds,
                    blur_score,
                )
                return

            empty_score = _frame_empty_score(frame)
            if empty_score < VIDEO_EMPTY_STD_THRESHOLD:
                empty_frames_skipped += 1
                logger.debug(
                    "Video OCR skipped empty frame %d at %.2fs (std=%.2f)",
                    frame_index,
                    timestamp_seconds,
                    empty_score,
                )
                return

            try:
                ocr_frame = _preprocess_video_frame(frame, aggressive=False)
                frame_text, confidence = _ocr_image(
                    ocr_frame,
                    preprocess=False,
                    psm=6,
                )

                if _should_retry_ocr(frame_text, confidence):
                    retry_count += 1
                    retry_frame = _preprocess_video_frame(frame, aggressive=True)
                    retry_text, retry_confidence = _ocr_image(
                        retry_frame,
                        preprocess=False,
                        psm=11,
                    )
                    if _prefer_ocr_candidate(
                        frame_text,
                        confidence,
                        retry_text,
                        retry_confidence,
                    ):
                        frame_text = retry_text
                        confidence = retry_confidence

                if _append_unique_lines(frame_text):
                    logger.debug(
                        "Video OCR accepted frame %d at %.2fs (conf=%.2f, text_len=%d)",
                        frame_index,
                        timestamp_seconds,
                        confidence,
                        len(frame_text.strip()),
                    )
                else:
                    logger.debug(
                        "Video OCR produced duplicate or short text for frame %d at %.2fs (conf=%.2f, text_len=%d)",
                        frame_index,
                        timestamp_seconds,
                        confidence,
                        len(frame_text.strip()),
                    )
            except Exception as exc:
                ocr_failures += 1
                logger.warning(
                    "Video OCR failed on frame %d at %.2fs: %s",
                    frame_index,
                    timestamp_seconds,
                    exc,
                )

        if fps > 0 and duration_seconds > 0:
            max_samples = min(
                VIDEO_MAX_SAMPLES,
                max(1, int(np.ceil(duration_seconds / frame_interval_seconds)) + 1),
            )
            if max_samples > 1:
                sample_times = np.linspace(0.0, duration_seconds, num=max_samples, endpoint=True)
            else:
                sample_times = np.array([0.0])

            for frame_index, timestamp_seconds in enumerate(sample_times):
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
                success, frame = capture.read()
                if not success:
                    continue
                total_frames_processed += 1
                _process_frame(frame, frame_index, float(timestamp_seconds))
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
                _process_frame(frame, attempts, timestamp_seconds)
                attempts += 1
                timestamp_seconds += frame_interval_seconds

        if not parts:
            message = "No meaningful OCR text extracted from video."
            _notify_streamlit_error(message)
            raise RuntimeError(message)

        extracted_text = "\n\n".join(parts)
        logger.debug(
            "Video OCR stats for %s: total_frames_processed=%d, sampled_frames=%d, frames_kept=%d, blurry_frames_skipped=%d, empty_frames_skipped=%d, ocr_failures=%d, retry_count=%d, extracted_text_length=%d",
            video_path,
            total_frames_processed,
            sampled_frames,
            frames_kept,
            blurry_frames_skipped,
            empty_frames_skipped,
            ocr_failures,
            retry_count,
            len(extracted_text),
        )

        return extracted_text
    finally:
        capture.release()


# ── Public API ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".docx"}


@traceable(name="OCR Extraction", run_type="tool")
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
            img.verify()  # detect corrupt images early
            img = Image.open(io.BytesIO(file_bytes))  # re-open after verify

            try:
                text, confidence = _ocr_image(img, preprocess=True, aggressive=False, psm=6)
                if _is_meaningful_text(text, MIN_IMAGE_TEXT_LENGTH):
                    return text

                retry_text, retry_confidence = _ocr_image(
                    img,
                    preprocess=True,
                    aggressive=True,
                    psm=11,
                )
                if _prefer_ocr_candidate(text, confidence, retry_text, retry_confidence):
                    text = retry_text

                if text.strip():
                    return text

                message = "No meaningful OCR text extracted from image."
                _notify_streamlit_error(message)
                raise RuntimeError(message)
            except Exception as exc:
                message = f"Image OCR failed: {exc}"
                _notify_streamlit_error(message)
                raise RuntimeError(message) from exc

        if ext == ".pdf":
            return _extract_pdf(file_bytes)

        if ext == ".docx":
            return _extract_docx(file_bytes)

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Text extraction error ({ext}): {exc}") from exc

    return ""