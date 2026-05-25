"""Production logging with rotating logs and secret masking."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import get_settings, mask_secret

_LOGGING_CONFIGURED = False

_SECRET_PATTERNS = [
    re.compile(r"(api[_ -]?key\s*[:=]\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"(authorization\s*[:=]\s*bearer\s+)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"(langchain_api_key\s*[:=]\s*)([^\s,;]+)", re.IGNORECASE),
]


class SecretMaskingFilter(logging.Filter):
    """Mask secrets in log messages before they are written."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in _SECRET_PATTERNS:
            message = pattern.sub(lambda match: f"{match.group(1)}{mask_secret(match.group(2))}", message)
        record.msg = message
        record.args = ()
        return True


class CategoryFilter(logging.Filter):
    def __init__(self, category: str):
        super().__init__()
        self.category = category

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "category", None) == self.category


def _build_handler(log_file: Path, level: int, category: Optional[str] = None) -> RotatingFileHandler:
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    handler.setFormatter(formatter)
    handler.addFilter(SecretMaskingFilter())
    if category is not None:
        handler.addFilter(CategoryFilter(category))
    return handler


def setup_logging() -> None:
    """Configure rotating logs once per process."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    settings = get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    root_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Remove default handlers if the app is reloaded.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    general_handler = _build_handler(settings.log_dir / "app.log", root_level)
    error_handler = _build_handler(settings.log_dir / "error.log", logging.ERROR)
    request_handler = _build_handler(settings.log_dir / "request.log", root_level, category="request")
    ocr_handler = _build_handler(settings.log_dir / "ocr.log", root_level, category="ocr")
    provider_handler = _build_handler(settings.log_dir / "provider.log", root_level, category="provider")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(root_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    console_handler.addFilter(SecretMaskingFilter())

    for handler in [general_handler, error_handler, request_handler, ocr_handler, provider_handler, console_handler]:
        root_logger.addHandler(handler)

    logging.getLogger("app.request").propagate = True
    logging.getLogger("app.ocr").propagate = True
    logging.getLogger("app.provider").propagate = True
    logging.getLogger("app.error").propagate = True

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger for the application."""
    setup_logging()
    return logging.getLogger(name)


def log_request(message: str, **fields) -> None:
    logger = logging.getLogger("app.request")
    logger.info(message, extra={"category": "request", **fields})


def log_ocr(message: str, **fields) -> None:
    logger = logging.getLogger("app.ocr")
    logger.info(message, extra={"category": "ocr", **fields})


def log_provider(message: str, **fields) -> None:
    logger = logging.getLogger("app.provider")
    logger.info(message, extra={"category": "provider", **fields})


def log_error(message: str, **fields) -> None:
    logger = logging.getLogger("app.error")
    logger.error(message, extra={"category": "error", **fields})
