"""Lightweight multimodal document analyzer."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config import get_settings, secure_filename
from download_utils import to_docx, to_pdf, to_txt
from llm_utils import (
    GeminiAPIError,
    PROVIDER_MODELS,
    SUMMARY_MODES,
    TONES,
    answer_with_context,
    analyze_resume_text,
    summarize_text,
)
from logging_utils import log_error, log_ocr, log_provider, setup_logging
from ocr_utils import VIDEO_EXTENSIONS, extract_text, extract_video_text
from rag_utils import build_index_from_documents, format_sources, retrieve
from tracing_utils import bootstrap_langsmith_tracing

SETTINGS = get_settings()
setup_logging()
bootstrap_langsmith_tracing()

MAX_UPLOAD_MB = SETTINGS.max_upload_mb

st.set_page_config(
    page_title="Multimodal Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #0b1020;
        color: #e5eefc;
    }
    [data-testid="stSidebar"] {
        background: #11182a;
        border-right: 1px solid #22304b;
    }
    .card {
        background: rgba(16, 20, 40, 0.96);
        border: 1px solid #23314d;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .banner-ok {
        background: #0b2f22;
        border-left: 4px solid #34d399;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .banner-err {
        background: #2d1015;
        border-left: 4px solid #f87171;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }
    textarea, input[type="text"], input[type="password"] {
        background: #11182a !important;
        color: #e5eefc !important;
        border-color: #2f3f60 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS: dict[str, Any] = {
    "provider": "Gemini",
    "model": PROVIDER_MODELS["Gemini"][0],
    "api_key": "",
    "summary_mode": SUMMARY_MODES[0],
    "tone": TONES[0],
    "summary": "",
    "resume_analysis": "",
    "extracted_text": "",
    "rag_collection": None,
    "chat_history": [],
    "uploaded_documents": [],
    "processed_file_hashes": set(),
    "current_file_name": "",
    "memory_turns": 4,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _check_rate_limit(action: str) -> bool:
    event_store = st.session_state.setdefault("rate_limit_events", {})
    now = time.time()
    timestamps = [timestamp for timestamp in event_store.get(action, []) if now - timestamp < 60]
    if len(timestamps) >= SETTINGS.rate_limit_burst:
        st.error("Rate limit reached. Please wait a moment and try again.")
        return False
    timestamps.append(now)
    event_store[action] = timestamps
    return True


def _rebuild_rag_index() -> None:
    documents = st.session_state["uploaded_documents"]
    if not documents:
        st.session_state["rag_collection"] = None
        return

    payload = [
        {
            "text": document["text"],
            "filename": document["filename"],
            "upload_time": document["upload_time"],
            "chunk_source": document["filename"],
            "page_number": None,
        }
        for document in documents
    ]

    try:
        with st.spinner("Building retrieval index..."):
            st.session_state["rag_collection"] = build_index_from_documents(payload)
    except Exception as exc:
        log_error(f"RAG index build failed: {exc}")
        st.error(f"RAG index build failed: {exc}")
        raise


def _render_source_cards(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander("Sources", expanded=False):
        for source in sources:
            citation = source.get("citation", "?")
            filename = source.get("filename", "unknown")
            page_no = source.get("page_number")
            page_label = f"Page {page_no}" if isinstance(page_no, int) and page_no > 0 else "Page n/a"
            st.markdown(f"**[{citation}] {filename}** · {page_label}")
            st.caption(source.get("text", ""))


def _render_chat_history() -> None:
    for message in st.session_state["chat_history"]:
        role = "user" if message.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message.get("text", ""))
            if role == "assistant":
                _render_source_cards(message.get("sources", []))


def _process_uploaded_file(uploaded_file) -> dict[str, Any]:
    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"{uploaded_file.name} exceeds the {MAX_UPLOAD_MB} MB limit.")

    file_hash = _file_hash(file_bytes)
    file_ext = os.path.splitext(uploaded_file.name.lower())[1]
    safe_name = secure_filename(uploaded_file.name)

    if file_ext not in {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"}:
        raise ValueError(f"Unsupported file type: {file_ext}")

    if file_hash in st.session_state["processed_file_hashes"]:
        return {
            "filename": safe_name,
            "file_hash": file_hash,
            "text": "",
            "type": file_ext,
            "duplicate": True,
        }

    if file_ext in VIDEO_EXTENSIONS:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext or ".mp4") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            text = extract_video_text(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        text = extract_text(file_bytes, uploaded_file.name)

    return {
        "filename": safe_name,
        "file_hash": file_hash,
        "text": text,
        "type": file_ext,
        "duplicate": False,
    }


def _load_uploads(uploaded_files) -> None:
    if not uploaded_files:
        return

    progress = st.progress(0.0)
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        progress.progress(index / len(uploaded_files))
        try:
            processed = _process_uploaded_file(uploaded_file)
            if processed["duplicate"]:
                st.info(f"Skipping already processed file: {uploaded_file.name}")
                continue

            st.session_state["uploaded_documents"].append(
                {
                    "filename": processed["filename"],
                    "file_hash": processed["file_hash"],
                    "text": processed["text"],
                    "type": processed["type"],
                    "upload_time": _utc_now(),
                }
            )
            st.session_state["processed_file_hashes"].add(processed["file_hash"])
            st.session_state["current_file_name"] = processed["filename"]
            st.session_state["extracted_text"] = processed["text"]
            st.session_state["summary"] = ""
            st.session_state["resume_analysis"] = ""
            log_ocr(
                f"OCR extraction completed for {processed['filename']}",
                category="ocr",
                file_type=processed["type"],
                file_hash=processed["file_hash"],
            )
            st.success(f"Extracted {len(processed['text']):,} characters from {uploaded_file.name}")
        except Exception as exc:
            log_error(f"OCR extraction failed for {uploaded_file.name}: {exc}")
            st.error(f"Extraction failed for {uploaded_file.name}: {exc}")

    progress.progress(1.0)

    try:
        _rebuild_rag_index()
    except Exception:
        st.warning("The document set was processed, but the retrieval index could not be rebuilt.")


def _export_summary_buttons(summary_text: str) -> None:
    st.download_button("Download TXT", data=to_txt(summary_text), file_name="document_summary.txt", mime="text/plain", use_container_width=True)
    st.download_button("Download PDF", data=to_pdf(summary_text), file_name="document_summary.pdf", mime="application/pdf", use_container_width=True)
    st.download_button("Download DOCX", data=to_docx(summary_text), file_name="document_summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)


def _summary_tab() -> None:
    st.markdown("### Upload document")
    st.caption("Upload a PDF, image, DOCX, or video for OCR and summary generation.")

    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, JPG, PNG, MP4, MOV, AVI",
        accept_multiple_files=True,
        type=["pdf", "docx", "jpg", "jpeg", "png", "mp4", "mov", "avi"],
        key="summary_uploader",
    )

    if uploaded_files:
        _load_uploads(uploaded_files)

    if st.session_state["extracted_text"]:
        st.markdown("### Extracted text")
        st.text_area("OCR output", st.session_state["extracted_text"], height=220, disabled=True)
        st.download_button(
            "Download extracted text",
            data=st.session_state["extracted_text"].encode("utf-8"),
            file_name="extracted_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("### Generate Summary")
        st.caption(
            f"Provider: {st.session_state['provider']} · Model: {st.session_state['model']} · Summary mode: {st.session_state['summary_mode']} · Tone: {st.session_state['tone']}"
        )

        if st.button("Generate Summary", use_container_width=True):
            if not st.session_state["api_key"]:
                st.warning(f"Enter your {st.session_state['provider']} API key in the sidebar.")
            elif not _check_rate_limit("summary"):
                st.warning("Summary request was rate-limited. Please wait a moment and retry.")
            else:
                try:
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(
                            st.session_state["extracted_text"],
                            provider=st.session_state["provider"],
                            model=st.session_state["model"],
                            api_key=st.session_state["api_key"],
                            mode=st.session_state["summary_mode"],
                            tone=st.session_state["tone"],
                        )
                    st.session_state["summary"] = summary
                    log_provider(
                        f"Summary completed with {st.session_state['provider']} / {st.session_state['model']}",
                        category="provider",
                    )
                    st.success("Summary generated.")
                except GeminiAPIError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    log_error(f"Summary generation failed: {exc}")
                    st.error(f"Summary failed: {exc}")

    if st.session_state["summary"]:
        st.markdown("### Summary")
        st.write(st.session_state["summary"])
        _export_summary_buttons(st.session_state["summary"])
    elif st.session_state["extracted_text"]:
        st.info("Use Generate Summary to create a concise summary of the current document.")


def _rag_tab() -> None:
    st.markdown("### RAG Chat")
    if not st.session_state["uploaded_documents"]:
        st.info("Upload one or more documents in the Summarize tab to enable retrieval and Q&A.")
        return

    if st.session_state["rag_collection"] is None:
        try:
            _rebuild_rag_index()
            st.success("Retrieval index is ready.")
        except Exception:
            st.stop()

    _render_chat_history()

    question = st.chat_input("Ask a question about the uploaded documents")
    if not question:
        return

    if not st.session_state["api_key"]:
        st.warning(f"Enter your {st.session_state['provider']} API key in the sidebar.")
        return

    if not _check_rate_limit("chat"):
        return

    st.session_state["chat_history"].append({"role": "user", "text": question.strip(), "created_at": _utc_now()})

    try:
        with st.spinner("Searching documents and generating a response..."):
            retrieved = retrieve(st.session_state["rag_collection"], question.strip(), k=5)
            sources = format_sources(retrieved)
            answer = answer_with_context(
                question.strip(),
                retrieved,
                provider=st.session_state["provider"],
                model=st.session_state["model"],
                api_key=st.session_state["api_key"],
                conversation_history=st.session_state["chat_history"],
                memory_turns=int(st.session_state["memory_turns"]),
            )
    except GeminiAPIError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        log_error(f"Chat response failed: {exc}")
        st.error(f"Chat response failed: {exc}")
        return

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "text": answer,
            "sources": sources,
            "created_at": _utc_now(),
        }
    )
    st.rerun()


def _resume_tab() -> None:
    st.markdown("### Resume Analyzer")
    if not st.session_state["extracted_text"]:
        st.info("Upload a resume in the Summarize tab to analyze it.")
        return

    job_desc = st.text_area("Job description (optional)", height=130, placeholder="Paste a job description to compare against the resume.")

    if st.button("Analyze Resume", use_container_width=True):
        if not st.session_state["api_key"]:
            st.warning(f"Enter your {st.session_state['provider']} API key in the sidebar.")
        elif not _check_rate_limit("resume"):
            st.warning("Resume analysis request was rate-limited. Please wait a moment and retry.")
        else:
            try:
                with st.spinner("Analyzing resume..."):
                    analysis = analyze_resume_text(
                        st.session_state["extracted_text"],
                        provider=st.session_state["provider"],
                        model=st.session_state["model"],
                        api_key=st.session_state["api_key"],
                        job_desc=job_desc,
                    )
                st.session_state["resume_analysis"] = analysis
                log_provider(
                    f"Resume analysis completed with {st.session_state['provider']} / {st.session_state['model']}",
                    category="provider",
                )
                st.success("Resume analyzed.")
            except GeminiAPIError as exc:
                st.error(str(exc))
            except Exception as exc:
                log_error(f"Resume analysis failed: {exc}")
                st.error(f"Analysis failed: {exc}")

    if st.session_state["resume_analysis"]:
        st.markdown("### Analysis")
        st.write(st.session_state["resume_analysis"])
        st.download_button(
            "Download analysis",
            data=to_txt(st.session_state["resume_analysis"]),
            file_name="resume_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _sidebar() -> None:
    st.sidebar.markdown("### Provider")
    provider = st.sidebar.selectbox(
        "Provider",
        options=list(PROVIDER_MODELS.keys()),
        index=list(PROVIDER_MODELS.keys()).index(st.session_state["provider"]),
    )
    st.session_state["provider"] = provider
    st.session_state["model"] = st.sidebar.selectbox(
        "Model",
        options=PROVIDER_MODELS[provider],
        index=PROVIDER_MODELS[provider].index(st.session_state["model"]) if st.session_state["model"] in PROVIDER_MODELS[provider] else 0,
    )
    st.session_state["api_key"] = st.sidebar.text_input("API key", type="password", value=st.session_state["api_key"])

    st.sidebar.markdown("### Settings")
    st.session_state["summary_mode"] = st.sidebar.selectbox("Summary mode", SUMMARY_MODES, index=SUMMARY_MODES.index(st.session_state["summary_mode"]))
    st.session_state["tone"] = st.sidebar.selectbox("Tone", TONES, index=TONES.index(st.session_state["tone"]))
    st.sidebar.caption("LangSmith tracing is enabled for OCR, summaries, retrieval, chat, and resume analysis.")


st.markdown(
    """
    <h1 style='text-align:center; margin-bottom: 0.2rem;'>Multimodal Document Analyzer</h1>
    <p style='text-align:center; color:#94a3b8; margin-bottom: 1rem;'>OCR, summarization, RAG chat, and resume analysis in a single lightweight workflow.</p>
    """,
    unsafe_allow_html=True,
)

_sidebar()

(summary_tab, rag_tab, resume_tab) = st.tabs(["Summarize", "RAG Chat", "Resume Analyzer"])

with summary_tab:
    _summary_tab()

with rag_tab:
    _rag_tab()

with resume_tab:
    _resume_tab()
