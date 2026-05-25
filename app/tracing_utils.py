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


def langsmith_tracing_enabled() -> bool:
    """Return True only when LangSmith is configured and explicitly enabled."""
    return bool(
        _langsmith_traceable is not None
        and Client is not None
        and _is_truthy(os.getenv("LANGCHAIN_TRACING_V2"))
        and os.getenv("LANGCHAIN_ENDPOINT")
        and os.getenv("LANGCHAIN_API_KEY")
        and os.getenv("LANGCHAIN_PROJECT")
    )


def _print_startup_diagnostics(enabled: bool) -> None:
    project_name = os.getenv("LANGCHAIN_PROJECT") or "GENAI"
    endpoint = os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
    api_key_status = _mask_api_key(os.getenv("LANGCHAIN_API_KEY"))
    print(
        "[LangSmith] "
        f"tracing_enabled={enabled} "
        f"project={project_name} "
        f"api_key={api_key_status} "
        f"endpoint={endpoint}",
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

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "GENAI"

    diagnostics: dict[str, Any] = {
        "enabled": False,
        "project": os.getenv("LANGCHAIN_PROJECT") or "GENAI",
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com",
        "api_key_detected": bool(os.getenv("LANGCHAIN_API_KEY")),
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
