"""
llm_utils.py — LLM provider wrappers with retry + fallback logic.

Providers
---------
- Gemini  : google-generativeai SDK (direct)
- OpenAI  : openai>=1.40 SDK
- Groq    : OpenAI-compatible via base_url=https://api.groq.com/openai/v1
- Claude  : anthropic SDK

No deprecated SDK syntax. No LangChain runtime dependency.
API keys: passed explicitly or read from environment variables.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

logger = logging.getLogger(__name__)

# ── Allowed Gemini models ───────────────────────────────────────────────────

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# ── Allowed models (current, non-deprecated) ─────────────────────────────────

PROVIDER_MODELS: dict[str, List[str]] = {
    "Gemini": GEMINI_MODELS,
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
    "Groq": [
        "llama-3.3-70b-versatile",   # primary — fast & capable
        "llama-3.1-8b-instant",      # fallback — very fast
        "mixtral-8x7b-32768",        # fallback — large context
    ],
    "Claude": [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
    ],
}

SUMMARY_MODES = ["Concise", "Detailed", "Bullet Points", "Executive", "Technical"]
TONES = ["Neutral", "Formal", "Casual", "Academic"]

# ── Key helper ────────────────────────────────────────────────────────────────

def _get_key(env_var: str, explicit: Optional[str]) -> str:
    key = explicit or os.getenv(env_var, "")
    if not key:
        raise ValueError(
            f"No API key provided. Pass it explicitly or set {env_var}."
        )
    return key.strip()


# ── Retry decorator (shared) ──────────────────────────────────────────────────

def _make_retry():
    """3 attempts with exponential back-off for transient errors."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(text: str, mode: str = "Concise", tone: str = "Neutral") -> str:
    mode_instructions = {
        "Concise":     "Provide a concise 3-5 sentence summary highlighting the main points.",
        "Detailed":    "Provide a detailed, comprehensive summary covering all key points and supporting details.",
        "Bullet Points": "Summarise the document as a structured bullet-point list of key points.",
        "Executive":   "Write a one-paragraph executive summary suitable for senior stakeholders.",
        "Technical":   "Provide a technically accurate summary retaining domain-specific terminology.",
    }
    tone_instructions = {
        "Neutral":   "Use a neutral, objective tone.",
        "Formal":    "Use a formal, professional tone.",
        "Casual":    "Use a clear, conversational tone.",
        "Academic":  "Use an academic, scholarly tone.",
    }
    style = mode_instructions.get(mode, mode_instructions["Concise"])
    tone_str = tone_instructions.get(tone, tone_instructions["Neutral"])
    return (
        f"{style} {tone_str}\n\n"
        f"Document text:\n{text[:12000]}\n\nSummary:"
    )


# ── Gemini error classifier ───────────────────────────────────────────────────

def _classify_gemini_error(exc: Exception) -> str:
    """
    Map a raw google-generativeai exception to a clean, user-facing message.
    Covers: invalid key, expired key, quota exceeded, model not found,
    permission denied, and generic network failures.
    """
    msg = str(exc).lower()

    # Invalid / expired API key
    if any(k in msg for k in ("api_key_invalid", "api key not valid",
                               "invalid api key", "unauthenticated",
                               "401", "403")):
        return (
            "🔑 **Invalid or expired Gemini API key.**\n\n"
            "Please check your key at https://aistudio.google.com/app/apikey "
            "and paste the correct value in the sidebar."
        )

    # Quota / rate-limit
    if any(k in msg for k in ("quota", "rate limit", "resource_exhausted",
                               "429", "too many requests")):
        return (
            "⏳ **Gemini quota exceeded or rate limit hit.**\n\n"
            "Wait a moment and try again, or switch to a different model "
            "(e.g. gemini-2.5-flash) in the sidebar."
        )

    # Model not found / not available
    if any(k in msg for k in ("model not found", "not_found", "404",
                               "does not exist", "model_not_found")):
        return (
            "🤖 **Gemini model not found.**\n\n"
            f"The selected model is unavailable. "
            "Try **gemini-2.5-flash** or **gemini-2.5-pro** from the sidebar."
        )

    # Permission / billing
    if any(k in msg for k in ("permission_denied", "billing", "disabled",
                               "access", "forbidden")):
        return (
            "🚫 **Gemini API access denied.**\n\n"
            "Ensure billing is enabled on your Google Cloud project "
            "and the Generative Language API is activated."
        )

    # Network / connection
    if any(k in msg for k in ("connection", "timeout", "network", "dns",
                               "unreachable", "socket")):
        return (
            "🌐 **Network error connecting to Gemini.**\n\n"
            "Check your internet connection and try again."
        )

    # Fallback — show cleaned message without raw traceback
    return f"⚠️ **Gemini error:** {str(exc)}"


# ── Core generate function (exact requested implementation) ───────────────────

import google.generativeai as genai


def generate_gemini_response(prompt, api_key, model="gemini-2.5-flash"):
    try:
        genai.configure(api_key=api_key)

        model_obj = genai.GenerativeModel(model)

        response = model_obj.generate_content(prompt)

        return response.text

    except Exception as e:
        raise Exception(f"Gemini API Error: {str(e)}")


class GeminiAPIError(Exception):
    """Clean, user-facing Gemini error (no raw traceback)."""


# ── Provider implementations ──────────────────────────────────────────────────

def _summarize_gemini(text: str, api_key: str, model: str,
                      mode: str, tone: str) -> str:
    """
    Summarise *text* via Gemini with retry and clean error messages.
    Uses generate_gemini_response() for all API calls.
    """
    prompt = _build_prompt(text, mode, tone)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _call():
        return generate_gemini_response(prompt, api_key, model)

    try:
        return _call()
    except Exception as exc:
        raise GeminiAPIError(_classify_gemini_error(exc)) from exc


def _summarize_openai(text: str, api_key: str, model: str,
                      mode: str, tone: str) -> str:
    from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

    client = OpenAI(api_key=api_key, timeout=30.0)
    prompt = _build_prompt(text, mode, tone)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
        reraise=True,
    )
    def _call():
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful document summariser."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    return _call()


def _summarize_groq(text: str, api_key: str, model: str,
                    mode: str, tone: str) -> str:
    """
    Groq via OpenAI-compatible endpoint.
    Falls back through PROVIDER_MODELS['Groq'] on model errors.
    """
    from openai import OpenAI, APIConnectionError, APITimeoutError

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=30.0,
    )
    prompt = _build_prompt(text, mode, tone)
    models_to_try = [model] + [
        m for m in PROVIDER_MODELS["Groq"] if m != model
    ]

    last_exc: Exception = RuntimeError("Groq: all models failed.")
    for try_model in models_to_try:
        try:
            @retry(
                stop=stop_after_attempt(2),
                wait=wait_exponential(multiplier=1, min=1, max=5),
                retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
                reraise=True,
            )
            def _call(m=try_model):
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": "You are a helpful document summariser."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content or ""

            result = _call()
            if try_model != model:
                logger.warning("Groq: fell back from '%s' to '%s'", model, try_model)
            return result
        except Exception as exc:
            logger.warning("Groq model '%s' failed: %s", try_model, exc)
            last_exc = exc

    raise last_exc


def _summarize_claude(text: str, api_key: str, model: str,
                      mode: str, tone: str) -> str:
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(text, mode, tone)

    @_make_retry()
    def _call():
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text if msg.content else ""

    return _call()


# ── Unified entry point ───────────────────────────────────────────────────────

def summarize_text(
    text: str,
    *,
    provider: str = "Gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    mode: str = "Concise",
    tone: str = "Neutral",
) -> str:
    """
    Summarise *text* using the chosen provider.

    Parameters
    ----------
    text     : Document text to summarise.
    provider : 'Gemini' | 'OpenAI' | 'Groq' | 'Claude'
    model    : Model override; defaults to first in PROVIDER_MODELS[provider].
    api_key  : Explicit key; falls back to environment variable.
    mode     : Summary style (Concise / Detailed / Bullet Points / Executive / Technical).
    tone     : Tone (Neutral / Formal / Casual / Academic).
    """
    if not text.strip():
        raise ValueError("No text provided for summarisation.")

    provider = provider.strip()
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(PROVIDER_MODELS)}")

    chosen_model = (model or PROVIDER_MODELS[provider][0]).strip()

    _env = {
        "Gemini": "GOOGLE_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Groq":   "GROQ_API_KEY",
        "Claude": "ANTHROPIC_API_KEY",
    }
    key = _get_key(_env[provider], api_key)

    dispatch = {
        "Gemini": _summarize_gemini,
        "OpenAI": _summarize_openai,
        "Groq":   _summarize_groq,
        "Claude": _summarize_claude,
    }
    return dispatch[provider](text, key, chosen_model, mode, tone)


# ── RAG / chat helpers ────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping word-level chunks for embedding."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def answer_with_context(
    question: str,
    context_chunks: List[str],
    provider: str,
    model: str,
    api_key: str,
) -> str:
    """Generate a grounded answer from retrieved RAG context chunks."""
    if not context_chunks:
        return "I could not find relevant context in the document to answer that question."

    context = "\n\n---\n\n".join(context_chunks[:5])
    prompt = (
        "You are a document assistant. Answer the question below using ONLY "
        "the provided document context. If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    # Reuse generate_gemini_response for Gemini; direct SDK for others
    if provider == "Gemini":
        try:
            return generate_gemini_response(prompt, api_key, model)
        except Exception as exc:
            raise GeminiAPIError(_classify_gemini_error(exc)) from exc

    elif provider in ("OpenAI", "Groq"):
        from openai import OpenAI
        base = "https://api.groq.com/openai/v1" if provider == "Groq" else None
        kwargs = {"api_key": api_key, "timeout": 30.0}
        if base:
            kwargs["base_url"] = base
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful document Q&A assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    elif provider == "Claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text if msg.content else ""

    return "Provider not supported for RAG chat."
