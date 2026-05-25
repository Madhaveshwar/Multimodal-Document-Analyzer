"""Centralized LangSmith tracing helpers with safe fallback behavior."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, TypeVar, cast

from dotenv import load_dotenv

load_dotenv(override=False)

try:
    import langchain  # noqa: F401
except Exception:
    langchain = None  # type: ignore[assignment]

try:
    from langsmith import traceable as _langsmith_traceable
except Exception:
    _langsmith_traceable = None

F = TypeVar("F", bound=Callable[..., Any])


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def langsmith_tracing_enabled() -> bool:
    """Return True only when LangSmith is configured and explicitly enabled."""
    return bool(
        _langsmith_traceable is not None
        and _is_truthy(os.getenv("LANGCHAIN_TRACING_V2"))
        and os.getenv("LANGCHAIN_ENDPOINT")
        and os.getenv("LANGCHAIN_API_KEY")
        and os.getenv("LANGCHAIN_PROJECT")
    )


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
