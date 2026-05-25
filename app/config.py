"""Centralized configuration for production and development modes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    environment: str
    debug: bool
    secret_key: str
    data_dir: Path
    log_dir: Path
    db_path: Path
    max_upload_mb: int
    session_timeout_minutes: int
    tesseract_cmd: str
    langchain_tracing_v2: bool
    langchain_endpoint: str
    langchain_api_key: str
    langchain_project: str
    rate_limit_per_minute: int
    rate_limit_burst: int
    streamlit_port: int
    health_port: int
    ocr_cache_size: int
    embedding_cache_size: int
    log_level: str

    @property
    def production(self) -> bool:
        return self.environment.lower() in {"prod", "production"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    environment = os.getenv("APP_ENV") or os.getenv("STREAMLIT_ENV") or "development"
    data_dir = Path(os.getenv("APP_DATA_DIR", DATA_DIR))
    log_dir = Path(os.getenv("APP_LOG_DIR", LOG_DIR))
    db_path = Path(os.getenv("APP_DB_PATH", data_dir / "app.sqlite3"))
    return Settings(
        environment=environment,
        debug=_get_bool("APP_DEBUG", default=not environment.lower() in {"prod", "production"}),
        secret_key=os.getenv("APP_SECRET_KEY", "change-me-in-production"),
        data_dir=data_dir,
        log_dir=log_dir,
        db_path=db_path,
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "15")),
        session_timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "60")),
        tesseract_cmd=os.getenv("TESSERACT_CMD", r"E:\GENAI\tesseract.exe"),
        langchain_tracing_v2=_get_bool("LANGCHAIN_TRACING_V2", default=False),
        langchain_endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY", ""),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "Multimodal-Document-Analyzer"),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
        rate_limit_burst=int(os.getenv("RATE_LIMIT_BURST", "120")),
        streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
        health_port=int(os.getenv("HEALTH_PORT", "8502")),
        ocr_cache_size=int(os.getenv("OCR_CACHE_SIZE", "128")),
        embedding_cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "1")),
        log_level=os.getenv("LOG_LEVEL", "INFO" if environment.lower() in {"prod", "production"} else "DEBUG"),
    )


def mask_secret(value: Optional[str], visible_tail: int = 4) -> str:
    """Return a masked representation of secrets for logs/UI."""
    if not value:
        return ""
    text = str(value)
    if len(text) <= visible_tail:
        return "*" * len(text)
    return f"{'*' * max(4, len(text) - visible_tail)}{text[-visible_tail:]}"


def secure_filename(filename: str) -> str:
    """Remove path traversal and unsafe characters from file names."""
    raw_name = str(filename or "uploaded_file").replace("\\", "/").split("/")[-1]
    cleaned = []
    for char in raw_name:
        if char.isalnum() or char in {".", "-", "_", " "}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    safe_name = "".join(cleaned).strip().strip(".")
    return safe_name or "uploaded_file"


def is_production() -> bool:
    return get_settings().production
