"""Centralized LangSmith tracing helpers with safe fallback behavior."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, TypeVar, cast

from dotenv import load_dotenv

load_dotenv(override=False)

try:
    from langsmith import Client
    from langsmith import traceable as _langsmith_traceable
except Exception:
    Client = None  # type: ignore[assignment]
    _langsmith_traceable = None

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_BOOTSTRAPPED = False
_STARTUP_TRACE_EMITTED = False


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _mask_api_key(value: Optional[str]) -> str:
    if not value:
        return "not detected"
    text = str(value)
    return f"detected ({text[:4]}...{text[-4:]})" if len(text) > 8 else "detected"


def _normalize_secret_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_streamlit_secrets() -> dict[str, str]:
    """Read LangSmith secrets from Streamlit if available."""
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return {}

    try:
        secrets = st.secrets
    except Exception:
        return {}

    try:
        data = secrets.to_dict()
    except Exception:
        try:
            data = dict(secrets)
        except Exception:
            return {}

    if not isinstance(data, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, value in data.items():
        normalized_value = _normalize_secret_value(value)
        if normalized_value is not None:
            normalized[str(key)] = normalized_value
    return normalized


def _resolve_langsmith_settings() -> dict[str, Any]:
    secrets = _read_streamlit_secrets()
    streamlit_cloud_mode = bool(secrets)

    resolved = {
        "LANGCHAIN_API_KEY": _normalize_secret_value(secrets.get("LANGCHAIN_API_KEY")) or os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": _normalize_secret_value(secrets.get("LANGCHAIN_PROJECT")) or os.getenv("LANGCHAIN_PROJECT") or "GENAI",
        "LANGCHAIN_TRACING_V2": _normalize_secret_value(secrets.get("LANGCHAIN_TRACING_V2")) or os.getenv("LANGCHAIN_TRACING_V2") or "true",
        "LANGCHAIN_ENDPOINT": _normalize_secret_value(secrets.get("LANGCHAIN_ENDPOINT")) or os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com",
    }

    if not streamlit_cloud_mode:
        resolved["LANGCHAIN_PROJECT"] = resolved["LANGCHAIN_PROJECT"] or "GENAI"
        resolved["LANGCHAIN_TRACING_V2"] = resolved["LANGCHAIN_TRACING_V2"] or "true"
        resolved["LANGCHAIN_ENDPOINT"] = resolved["LANGCHAIN_ENDPOINT"] or "https://api.smith.langchain.com"

    for key, value in resolved.items():
        if value is not None:
            os.environ[key] = str(value)

    return {
        "streamlit_cloud_mode": streamlit_cloud_mode,
        "secrets_detected": bool(secrets),
        **resolved,
    }


def langsmith_tracing_enabled() -> bool:
    """Return True only when LangSmith is configured and explicitly enabled."""
    settings = _resolve_langsmith_settings()
    return bool(
        _langsmith_traceable is not None
        and Client is not None
        and _is_truthy(settings["LANGCHAIN_TRACING_V2"])
        and settings["LANGCHAIN_ENDPOINT"]
        and settings["LANGCHAIN_API_KEY"]
        and settings["LANGCHAIN_PROJECT"]
    )


def _print_startup_diagnostics(enabled: bool) -> None:
    settings = _resolve_langsmith_settings()
    project_name = settings["LANGCHAIN_PROJECT"] or "GENAI"
    endpoint = settings["LANGCHAIN_ENDPOINT"] or "https://api.smith.langchain.com"
    api_key_status = _mask_api_key(settings["LANGCHAIN_API_KEY"])
    print(
        "[LangSmith] "
        f"tracing_enabled={enabled} "
        f"project={project_name} "
        f"api_key={api_key_status} "
        f"endpoint={endpoint} "
        f"secrets_detected={settings['secrets_detected']} "
        f"streamlit_cloud_mode={settings['streamlit_cloud_mode']}",
        flush=True,
    )


def _startup_trace_check() -> None:
    global _STARTUP_TRACE_EMITTED
    if _STARTUP_TRACE_EMITTED:
        return

    @traceable(name="startup_trace_check", run_type="tool")
    def _emit_startup_trace() -> str:
        return "startup-trace-ok"

    try:
        _emit_startup_trace()
    except Exception as exc:
        logger.debug("LangSmith startup trace suppressed: %s", exc)
    finally:
        _STARTUP_TRACE_EMITTED = True


def bootstrap_langsmith_tracing() -> dict[str, Any]:
    """Initialize LangSmith tracing once and emit safe startup diagnostics."""
    global _BOOTSTRAPPED

    settings = _resolve_langsmith_settings()
    diagnostics: dict[str, Any] = {
        "enabled": False,
        "project": settings["LANGCHAIN_PROJECT"],
        "endpoint": settings["LANGCHAIN_ENDPOINT"],
        "api_key_detected": bool(settings["LANGCHAIN_API_KEY"]),
        "secrets_detected": settings["secrets_detected"],
        "streamlit_cloud_mode": settings["streamlit_cloud_mode"],
    }

    enabled = langsmith_tracing_enabled()
    diagnostics["enabled"] = enabled
    _print_startup_diagnostics(enabled)

    if not enabled:
        logger.info("LangSmith tracing disabled or not fully configured.")
        _BOOTSTRAPPED = True
        return diagnostics

    try:
        Client()
    except Exception as exc:
        diagnostics["enabled"] = False
        diagnostics["error"] = str(exc)
        logger.warning("LangSmith client initialization failed: %s", exc)
        _print_startup_diagnostics(False)
        _BOOTSTRAPPED = True
        return diagnostics

    try:
        _startup_trace_check()
    except Exception as exc:
        diagnostics["startup_trace_error"] = str(exc)
        logger.debug("LangSmith startup trace check failed: %s", exc)

    _BOOTSTRAPPED = True
    return diagnostics


def traceable(
    *,
    name: Optional[str] = None,
    run_type: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> Callable[[F], F]:
    """LangSmith traceable decorator with a no-op fallback when disabled."""
    if not langsmith_tracing_enabled():
        def _identity(func: F) -> F:
            return func

        return _identity

    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if run_type is not None:
        kwargs["run_type"] = run_type
    if metadata is not None:
        kwargs["metadata"] = metadata
    if tags is not None:
        kwargs["tags"] = tags

    return cast(Callable[[F], F], _langsmith_traceable(**kwargs))
