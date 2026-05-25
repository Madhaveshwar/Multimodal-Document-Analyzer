"""
llm_utils.py — LLM provider wrappers with retry + fallback logic.

Providers
---------
- Gemini  : google-generativeai SDK (direct)
- OpenAI  : openai>=1.40 SDK
- Groq    : OpenAI-compatible via base_url=https://api.groq.com/openai/v1
- Claude  : anthropic SDK

API keys: passed explicitly or read from environment variables.
LangSmith tracing is optional and safely disabled when not configured.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional, List, Any, Sequence

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from tracing_utils import traceable

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


def _normalize_chat_history(chat_history: Optional[Sequence[dict]], max_turns: int) -> List[dict]:
    if not chat_history:
        return []

    turns: List[dict] = []
    for message in chat_history[-(max_turns * 2):]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower().strip()
        text = str(message.get("text", "")).strip()
        if role not in {"user", "assistant", "bot"} or not text:
            continue
        turns.append({"role": "assistant" if role == "bot" else role, "text": text})
    return turns


def _format_chat_history(chat_history: Optional[Sequence[dict]], max_turns: int) -> str:
    turns = _normalize_chat_history(chat_history, max_turns=max_turns)
    if not turns:
        return "No prior conversation."

    lines: List[str] = []
    for turn in turns:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def _format_source_chunks(source_chunks: Optional[Sequence[Any]], max_sources: int = 8) -> List[str]:
    if not source_chunks:
        return []

    formatted: List[str] = []
    for index, chunk in enumerate(source_chunks[:max_sources], start=1):
        if isinstance(chunk, dict):
            text = str(chunk.get("text", "")).strip()
            metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), dict) else {}
            filename = metadata.get("filename", "unknown")
            page_number = metadata.get("page_number", -1)
            chunk_source = metadata.get("chunk_source", filename)
            upload_time = metadata.get("upload_time", "unknown")
            citation = chunk.get("citation", index)
            score = chunk.get("score", 0.0)
            header = f"[{citation}] {filename}"
            if page_number not in (-1, None):
                header += f" p.{page_number}"
            header += f" | source={chunk_source} | uploaded={upload_time} | score={score:.3f}"
            formatted.append(f"{header}\n{text}")
        else:
            text = str(chunk).strip()
            if text:
                formatted.append(f"[{index}] {text}")
    return formatted


def _compose_grounded_prompt(
    question: str,
    context_chunks: Sequence[Any],
    conversation_history: Optional[Sequence[dict]] = None,
    memory_turns: int = 6,
) -> str:
    history_block = _format_chat_history(conversation_history, max_turns=memory_turns)
    source_blocks = _format_source_chunks(context_chunks)
    source_text = "\n\n---\n\n".join(source_blocks) if source_blocks else "No source chunks were retrieved."
    return (
        "You are a production document assistant. Answer the user's question using only the provided source chunks and the recent conversation context. "
        "If the answer is not supported by the sources, say that clearly. Use citations like [1], [2] in your response for factual claims.\n\n"
        f"Recent conversation:\n{history_block}\n\n"
        f"Source chunks:\n{source_text}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations and keep it concise but complete:"
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


@traceable(name="Gemini API Call", run_type="llm")
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

@traceable(name="Generate Summary", run_type="chain")
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


@traceable(name="OpenAI API Call", run_type="llm")
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


@traceable(name="Groq API Call", run_type="llm")
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


@traceable(name="Claude API Call", run_type="llm")
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

@traceable(name="Generate Summary", run_type="chain")
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


@traceable(name="LLM Response", run_type="chain")
def answer_with_context(
    question: str,
    context_chunks: List[Any],
    provider: str,
    model: str,
    api_key: str,
    conversation_history: Optional[Sequence[dict]] = None,
    memory_turns: int = 6,
) -> str:
    """Generate a grounded answer from retrieved RAG context chunks."""
    prompt = _compose_grounded_prompt(
        question,
        context_chunks[:8],
        conversation_history=conversation_history,
        memory_turns=memory_turns,
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


@traceable(name="Resume Analysis", run_type="chain")
def analyze_resume_text(
    resume_text: str,
    provider: str,
    model: str,
    api_key: str,
    job_desc: str = "",
) -> str:
    """Generate a resume analysis using the existing chat response flow."""
    base_prompt = (
        "You are an expert HR recruiter and resume analyst.\n"
        "Analyze the following resume and provide:\n"
        "1. **Strengths** — top 3 strong points\n"
        "2. **Weaknesses** — top 3 areas to improve\n"
        "3. **Key Skills** — list detected technical and soft skills\n"
        "4. **Experience Summary** — 2-3 sentences\n"
        "5. **ATS Score (0-100)** — estimated applicant tracking score\n"
    )
    if job_desc.strip():
        base_prompt += (
            "6. **Job Match %** — how well this resume matches the job description below\n"
            "7. **Gaps** — what's missing for the role\n\n"
            f"Job Description:\n{job_desc}\n\n"
        )
    base_prompt += f"\nResume:\n{resume_text[:8000]}"

    return answer_with_context(
        base_prompt,
        [resume_text[:8000]],
        provider=provider,
        model=model,
        api_key=api_key,
        conversation_history=None,
        memory_turns=1,
    )
