"""Eshwar — SaaS-style Multimodal Document Analyzer."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config import get_settings, mask_secret, secure_filename
from database import (
    authenticate_user,
    clear_chat_history,
    clear_workspace_embeddings,
    create_auth_session,
    create_user,
    create_workspace,
    delete_document,
    delete_workspace,
    end_auth_session,
    get_auth_session,
    get_user_by_id,
    get_workspace,
    init_db,
    list_documents,
    list_workspaces,
    log_event,
    rename_workspace,
    restore_workspace_payload,
    save_chat_message,
    save_document,
    save_document_chunks,
    touch_auth_session,
    update_workspace_activity,
    update_workspace_state,
)
from download_utils import chat_history_to_docx, chat_history_to_pdf, chat_history_to_txt, to_docx, to_pdf, to_txt
from llm_utils import (
    GeminiAPIError,
    PROVIDER_MODELS,
    SUMMARY_MODES,
    TONES,
    analyze_resume_text,
    answer_with_context,
    summarize_text,
)
from health_utils import get_health_report
from ocr_utils import VIDEO_EXTENSIONS, extract_text, extract_video_text
from logging_utils import log_error, log_ocr, log_provider, log_request, setup_logging
from rag_utils import (
    build_index_from_chunk_records,
    build_index_from_documents,
    embed_texts,
    format_sources,
    prepare_chunk_records_from_documents,
    retrieve,
)

SETTINGS = get_settings()
setup_logging()

SESSION_TIMEOUT_MINUTES = SETTINGS.session_timeout_minutes
MAX_UPLOAD_MB = SETTINGS.max_upload_mb
RATE_LIMIT_PER_MINUTE = SETTINGS.rate_limit_per_minute
RATE_LIMIT_BURST = SETTINGS.rate_limit_burst


st.set_page_config(
    page_title="Eshwar — Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()


st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top, #171b2b 0%, #0f1117 58%, #0b0d12 100%);
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14182a 0%, #101521 100%);
        border-right: 1px solid #263042;
    }
    .card {
        background: rgba(20, 24, 38, 0.9);
        border: 1px solid #2a3347;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    }
    .banner-ok {
        background: #0d3b2e;
        border-left: 4px solid #22c55e;
        padding: .6rem 1rem;
        border-radius: 8px;
        margin-bottom: .8rem;
    }
    .banner-err {
        background: #3b0d0d;
        border-left: 4px solid #ef4444;
        padding: .6rem 1rem;
        border-radius: 8px;
        margin-bottom: .8rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: .55rem 1.1rem;
    }
    .stButton > button:hover { opacity: .92; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #a78bfa;
        border-bottom: 2px solid #a78bfa;
    }
    textarea, input[type="text"], input[type="password"], input[type="email"] {
        background-color: #161b2b !important;
        color: #e5e7eb !important;
        border-color: #2d3448 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULTS: dict[str, Any] = {
    "authenticated": False,
    "auth_token": "",
    "current_user": None,
    "active_workspace_id": None,
    "provider": "Gemini",
    "model": PROVIDER_MODELS["Gemini"][0],
    "api_key": "",
    "summary_mode": SUMMARY_MODES[0],
    "tone": TONES[0],
    "rag_top_k": 5,
    "memory_turns": 6,
    "summary": "",
    "resume_analysis": "",
    "extracted_text": "",
    "rag_collection": None,
    "chat_history": [],
    "uploaded_documents": [],
    "processed_file_hashes": set(),
    "workspace_payload": {"workspace": None, "documents": [], "chunks": [], "chats": []},
    "new_workspace_name": "",
    "rename_workspace_name": "",
    "delete_workspace_confirm": False,
    "clear_embeddings_confirm": False,
    "last_active_ts": time.time(),
    "rate_limit_events": {},
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _estimate_tokens(*texts: str) -> int:
    words = 0
    for text in texts:
        words += len(str(text or "").split())
    return max(1, int(words * 1.3))


def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _check_rate_limit(action: str) -> bool:
    now = time.time()
    events = st.session_state.setdefault("rate_limit_events", {})
    timestamps = [timestamp for timestamp in events.get(action, []) if now - timestamp < 60]
    if len(timestamps) >= RATE_LIMIT_BURST or len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        st.error("Rate limit reached. Please wait a moment and try again.")
        return False
    timestamps.append(now)
    events[action] = timestamps
    return True


def _mark_activity() -> None:
    st.session_state["last_active_ts"] = time.time()
    auth_token = st.session_state.get("auth_token")
    workspace_id = st.session_state.get("active_workspace_id")
    current_user = st.session_state.get("current_user")
    if auth_token:
        try:
            touch_auth_session(auth_token)
        except Exception:
            pass
    if workspace_id:
        try:
            update_workspace_activity(int(workspace_id))
        except Exception:
            pass
    if current_user:
        try:
            log_event(int(current_user["id"]), workspace_id=workspace_id, event_type="activity", details={"source": "ui"})
        except Exception:
            pass


def _logout() -> None:
    auth_token = st.session_state.get("auth_token")
    if auth_token:
        try:
            end_auth_session(auth_token)
        except Exception:
            pass
    st.session_state["authenticated"] = False
    st.session_state["auth_token"] = ""
    st.session_state["current_user"] = None
    st.session_state["active_workspace_id"] = None
    st.session_state["summary"] = ""
    st.session_state["resume_analysis"] = ""
    st.session_state["extracted_text"] = ""
    st.session_state["rag_collection"] = None
    st.session_state["chat_history"] = []
    st.session_state["uploaded_documents"] = []
    st.session_state["processed_file_hashes"] = set()
    st.session_state["workspace_payload"] = {"workspace": None, "documents": [], "chunks": [], "chats": []}
    st.rerun()


def _save_workspace_state() -> None:
    workspace_id = st.session_state.get("active_workspace_id")
    if not workspace_id:
        return
    try:
        update_workspace_state(
            int(workspace_id),
            summary_text=st.session_state.get("summary") or "",
            last_document_text=st.session_state.get("extracted_text") or "",
            provider=st.session_state.get("provider"),
            model=st.session_state.get("model"),
        )
    except Exception:
        pass


def _persist_chat(role: str, text: str, sources: list[dict[str, Any]] | None = None, provider: str | None = None, latency_ms: float | None = None, token_usage: int | None = None) -> None:
    workspace_id = st.session_state.get("active_workspace_id")
    current_user = st.session_state.get("current_user")
    if not workspace_id or not current_user:
        return
    save_chat_message(
        int(workspace_id),
        role=role,
        text=text,
        sources=sources or [],
        provider=provider,
        latency_ms=latency_ms,
        token_usage=token_usage,
    )
    try:
        log_event(
            int(current_user["id"]),
            workspace_id=int(workspace_id),
            event_type="chat" if role == "assistant" else "user_message",
            provider=provider,
            token_usage=token_usage,
            latency_ms=latency_ms,
            details={"role": role},
        )
    except Exception:
        pass


def _persist_ocr_event(provider: str | None, latency_ms: float | None, file_name: str, file_type: str) -> None:
    current_user = st.session_state.get("current_user")
    workspace_id = st.session_state.get("active_workspace_id")
    if not current_user:
        return
    try:
        log_event(
            int(current_user["id"]),
            workspace_id=int(workspace_id) if workspace_id else None,
            event_type="ocr",
            provider=provider,
            latency_ms=latency_ms,
            details={"file_name": file_name, "file_type": file_type},
        )
    except Exception:
        pass


def _persist_llm_event(event_type: str, provider: str, latency_ms: float, prompt_text: str, response_text: str) -> None:
    current_user = st.session_state.get("current_user")
    workspace_id = st.session_state.get("active_workspace_id")
    if not current_user:
        return
    token_usage = _estimate_tokens(prompt_text, response_text)
    try:
        log_event(
            int(current_user["id"]),
            workspace_id=int(workspace_id) if workspace_id else None,
            event_type=event_type,
            provider=provider,
            token_usage=token_usage,
            latency_ms=latency_ms,
            details={"token_usage_mode": "estimated"},
        )
    except Exception:
        pass


def _render_source_cards(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("Sources used", expanded=False):
        for source in sources:
            filename = source.get("filename", "unknown")
            page_number = source.get("page_number")
            page_label = f"Page {page_number}" if isinstance(page_number, int) and page_number > 0 else "Page n/a"
            citation = source.get("citation", "?")
            score = float(source.get("score", 0.0))
            st.markdown(f"**[{citation}] {filename}** · {page_label} · source: {source.get('chunk_source', 'unknown')} · score: {score:.3f}")
            st.caption(source.get("text", ""))


def _render_chat_history() -> None:
    for message in st.session_state["chat_history"]:
        role = "user" if message.get("role") == "user" else "assistant"
        with st.chat_message(role):
            created_at = message.get("created_at")
            prefix = f"{created_at} · " if created_at else ""
            st.markdown(f"{prefix}{message.get('text', '')}")
            if role == "assistant":
                _render_source_cards(message.get("sources", []))


def _save_uploaded_document(file_name: str, file_bytes: bytes, text: str, ext: str) -> dict[str, Any]:
    workspace_id = int(st.session_state["active_workspace_id"])
    upload_time = _utc_now()
    document_row = save_document(
        workspace_id,
        filename=file_name,
        file_hash=_file_hash(file_bytes),
        upload_time=upload_time,
        mime_type=ext.lstrip(".").lower() or "unknown",
        size_bytes=len(file_bytes),
        source_type="video" if ext in VIDEO_EXTENSIONS else "ocr",
        text_content=text,
        metadata={
            "filename": file_name,
            "upload_time": upload_time,
            "chunk_source": file_name,
            "page_number": None,
            "size_bytes": len(file_bytes),
        },
    )
    if not document_row:
        raise RuntimeError("Unable to save document.")

    chunk_records = prepare_chunk_records_from_documents([
        {
            "text": text,
            "filename": file_name,
            "upload_time": upload_time,
            "chunk_source": file_name,
            "page_number": None,
        }
    ])
    embeddings = embed_texts([chunk["text"] for chunk in chunk_records]) if chunk_records else []
    for index, chunk in enumerate(chunk_records):
        chunk["embedding"] = embeddings[index] if index < len(embeddings) else None
    save_document_chunks(document_row["id"], workspace_id, chunk_records)
    return document_row


def _load_workspace(workspace_id: int) -> None:
    payload = restore_workspace_payload(int(workspace_id))
    workspace = payload["workspace"]
    st.session_state["workspace_payload"] = payload
    st.session_state["active_workspace_id"] = workspace["id"]
    st.session_state["uploaded_documents"] = payload["documents"]
    st.session_state["processed_file_hashes"] = {
        document.get("file_hash") for document in payload["documents"] if document.get("file_hash")
    }
    st.session_state["processed_file_hashes"] = {document.get("file_hash") for document in payload["documents"] if document.get("file_hash")}
    st.session_state["chat_history"] = [
        {
            "role": message["role"],
            "text": message["text"],
            "sources": message.get("sources", []),
            "created_at": message.get("created_at"),
        }
        for message in payload["chats"]
    ]
    st.session_state["summary"] = workspace.get("summary_text") or ""
    st.session_state["extracted_text"] = workspace.get("last_document_text") or (
        payload["documents"][0].get("text_content", "") if payload["documents"] else ""
    )
    if workspace.get("provider"):
        st.session_state["provider"] = workspace["provider"]
    if workspace.get("model"):
        st.session_state["model"] = workspace["model"]
    try:
        st.session_state["rag_collection"] = build_index_from_chunk_records(payload["chunks"]) if payload["chunks"] else None
    except Exception:
        st.session_state["rag_collection"] = build_index_from_documents(payload["documents"]) if payload["documents"] else None
    _mark_activity()


def _load_or_create_initial_workspace(user_id: int) -> None:
    workspaces = list_workspaces(user_id)
    if not workspaces:
        workspaces = [create_workspace(user_id, "My Workspace")]
    _load_workspace(int(workspaces[0]["id"]))


def _maybe_timeout_session() -> None:
    if not st.session_state["authenticated"]:
        return
    auth_session = get_auth_session(st.session_state["auth_token"])
    if not auth_session:
        _logout()
        return
    if time.time() - float(st.session_state.get("last_active_ts", 0)) > SESSION_TIMEOUT_MINUTES * 60:
        _logout()
        return
    _mark_activity()


def _auth_screen() -> None:
    st.markdown("<h1 style='text-align:center; color:#a78bfa; margin-top:1.5rem;'>Eshwar SaaS Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8;'>Secure document intelligence with saved workspaces, RAG, and analytics.</p>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        auth_tab_login, auth_tab_signup = st.tabs(["Login", "Sign up"])

        with auth_tab_login:
            with st.form("login_form"):
                identifier = st.text_input("Username or email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
            if submitted:
                user = authenticate_user(identifier, password)
                if not user:
                    st.error("Invalid credentials.")
                else:
                    session_info = create_auth_session(int(user["id"]))
                    st.session_state["authenticated"] = True
                    st.session_state["auth_token"] = session_info["token"]
                    st.session_state["current_user"] = user
                    _load_or_create_initial_workspace(int(user["id"]))
                    st.rerun()

        with auth_tab_signup:
            with st.form("signup_form"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account")
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    try:
                        user = create_user(username, email, password)
                        session_info = create_auth_session(int(user["id"]))
                        st.session_state["authenticated"] = True
                        st.session_state["auth_token"] = session_info["token"]
                        st.session_state["current_user"] = user
                        create_workspace(int(user["id"]), "My Workspace")
                        _load_or_create_initial_workspace(int(user["id"]))
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
        st.markdown("</div>", unsafe_allow_html=True)


def _sidebar_after_auth() -> None:
    user = st.session_state["current_user"]
    st.sidebar.markdown(f"### 👤 {user['username']}")
    st.sidebar.caption(user["email"])
    if st.sidebar.button("Logout", use_container_width=True):
        _logout()

    st.sidebar.markdown("---")
    workspaces = list_workspaces(int(user["id"]))
    if not workspaces:
        workspaces = [create_workspace(int(user["id"]), "My Workspace")]
    workspace_labels = [f"{workspace['name']} · #{workspace['id']}" for workspace in workspaces]
    workspace_id_map = {label: workspace["id"] for label, workspace in zip(workspace_labels, workspaces)}
    current_label = next((label for label, workspace_id in workspace_id_map.items() if workspace_id == st.session_state.get("active_workspace_id")), workspace_labels[0])
    selected_label = st.sidebar.selectbox("Workspace", workspace_labels, index=workspace_labels.index(current_label))
    selected_id = int(workspace_id_map[selected_label])
    if selected_id != int(st.session_state.get("active_workspace_id") or -1):
        _load_workspace(selected_id)
        st.rerun()

    st.sidebar.text_input("New workspace name", key="new_workspace_name", placeholder="Create a new workspace…")
    if st.sidebar.button("Create workspace", use_container_width=True):
        if st.session_state["new_workspace_name"].strip():
            workspace = create_workspace(int(user["id"]), st.session_state["new_workspace_name"])
            st.session_state["new_workspace_name"] = ""
            _load_workspace(int(workspace["id"]))
            st.rerun()

    workspace = get_workspace(int(st.session_state["active_workspace_id"]))
    st.sidebar.text_input("Rename workspace", key="rename_workspace_name", value=workspace["name"] if workspace else "")
    if st.sidebar.button("Rename workspace", use_container_width=True):
        if st.session_state["rename_workspace_name"].strip():
            rename_workspace(int(st.session_state["active_workspace_id"]), st.session_state["rename_workspace_name"])
            _load_workspace(int(st.session_state["active_workspace_id"]))
            st.rerun()

    st.sidebar.checkbox("Confirm delete workspace", key="delete_workspace_confirm")
    if st.sidebar.button("Delete workspace", use_container_width=True):
        if st.session_state["delete_workspace_confirm"]:
            delete_workspace(int(st.session_state["active_workspace_id"]))
            remaining = list_workspaces(int(user["id"]))
            if not remaining:
                remaining = [create_workspace(int(user["id"]), "My Workspace")]
            _load_workspace(int(remaining[0]["id"]))
            st.rerun()

    st.sidebar.checkbox("Confirm clear embeddings", key="clear_embeddings_confirm")
    if st.sidebar.button("Clear embeddings", use_container_width=True):
        if st.session_state["clear_embeddings_confirm"]:
            clear_workspace_embeddings(int(st.session_state["active_workspace_id"]))
            _load_workspace(int(st.session_state["active_workspace_id"]))
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ AI Provider")
    provider = st.sidebar.selectbox("Provider", options=list(PROVIDER_MODELS.keys()), index=list(PROVIDER_MODELS.keys()).index(st.session_state["provider"]))
    if provider != st.session_state["provider"]:
        st.session_state["provider"] = provider
        st.session_state["model"] = PROVIDER_MODELS[provider][0]
    st.session_state["model"] = st.sidebar.selectbox(
        "Model",
        options=PROVIDER_MODELS[provider],
        index=PROVIDER_MODELS[provider].index(st.session_state["model"]) if st.session_state["model"] in PROVIDER_MODELS[provider] else 0,
    )
    st.session_state["api_key"] = st.sidebar.text_input("API Key", type="password", value=st.session_state["api_key"])
    st.sidebar.markdown("### 📊 Generation Settings")
    st.session_state["summary_mode"] = st.sidebar.selectbox("Summary mode", SUMMARY_MODES, index=SUMMARY_MODES.index(st.session_state["summary_mode"]))
    st.session_state["tone"] = st.sidebar.selectbox("Tone", TONES, index=TONES.index(st.session_state["tone"]))
    st.session_state["rag_top_k"] = st.sidebar.slider("Top-k source chunks", min_value=3, max_value=10, value=int(st.session_state["rag_top_k"]), step=1)
    st.session_state["memory_turns"] = st.sidebar.slider("Memory turns", min_value=2, max_value=12, value=int(st.session_state["memory_turns"]), step=1)
    st.sidebar.caption("Session-based auth, persisted workspaces, and resumable chat history.")


def _workspace_overview() -> None:
    workspace = get_workspace(int(st.session_state["active_workspace_id"]))
    documents = list_documents(int(st.session_state["active_workspace_id"]))
    cols = st.columns(4)
    cols[0].metric("Documents", len(documents))
    cols[1].metric("Chat turns", len(st.session_state["chat_history"]))
    cols[2].metric("Summary", "Yes" if st.session_state.get("summary") else "No")
    cols[3].metric("Embeddings", "Loaded" if st.session_state.get("rag_collection") is not None else "None")
    st.caption(f"Workspace: {workspace['name']} · created {workspace['created_at']} · last opened {workspace.get('last_opened_at') or 'unknown'}")


def _export_chat_controls() -> None:
    messages = st.session_state["chat_history"]
    if not messages:
        st.info("No conversation to export yet.")
        return
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("⬇️ TXT", data=chat_history_to_txt(messages), file_name="conversation.txt", mime="text/plain", use_container_width=True)
    with e2:
        st.download_button("⬇️ PDF", data=chat_history_to_pdf(messages), file_name="conversation.pdf", mime="application/pdf", use_container_width=True)
    with e3:
        st.download_button("⬇️ DOCX", data=chat_history_to_docx(messages), file_name="conversation.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)


def _rebuild_workspace_index() -> None:
    payload = restore_workspace_payload(int(st.session_state["active_workspace_id"]))
    st.session_state["uploaded_documents"] = payload["documents"]
    st.session_state["processed_file_hashes"] = {
        document.get("file_hash") for document in payload["documents"] if document.get("file_hash")
    }
    chunk_records = payload["chunks"]
    if not chunk_records:
        st.session_state["rag_collection"] = None
        return
    with st.spinner("Building workspace embeddings…"):
        try:
            st.session_state["rag_collection"] = build_index_from_chunk_records(chunk_records)
        except Exception:
            st.session_state["rag_collection"] = build_index_from_documents(payload["documents"])


def _upload_documents_tab() -> None:
    st.markdown("### Upload documents")
    supported = ", ".join(sorted(set(["pdf", "docx", "jpg", "jpeg", "png", "mp4", "mov", "avi"])))
    uploaded_files = st.file_uploader(
        f"Supported formats: {supported}",
        type=["pdf", "docx", "jpg", "jpeg", "png", "mp4", "mov", "avi"],
        accept_multiple_files=True,
        key="uploader_sum",
    )
    if not uploaded_files:
        return

    progress = st.progress(0.0)
    for index, uploaded in enumerate(uploaded_files, start=1):
        progress.progress((index - 1) / max(1, len(uploaded_files)))
        file_bytes = uploaded.read()
        file_hash = _file_hash(file_bytes)
        if file_hash in st.session_state.get("processed_file_hashes", set()):
            st.info(f"Skipping already saved file: {uploaded.name}")
            continue
        if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"{uploaded.name} exceeds the {MAX_UPLOAD_MB} MB limit.")
            continue

        _, ext = os.path.splitext(uploaded.name.lower())
        safe_name = secure_filename(uploaded.name)
        if ext not in {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"}:
            st.error(f"Unsupported file type: {ext}")
            continue

        if not _check_rate_limit("upload"):
            break

        with st.status(f"Processing {uploaded.name}", expanded=False) as status:
            started = time.perf_counter()
            tmp_path = None
            try:
                if ext in VIDEO_EXTENSIONS:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".mp4") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    text = extract_video_text(tmp_path)
                else:
                    text = extract_text(file_bytes, uploaded.name)

                document_row = _save_uploaded_document(safe_name, file_bytes, text, ext)
                st.session_state.setdefault("processed_file_hashes", set()).add(file_hash)
                st.session_state["extracted_text"] = text
                st.session_state["summary"] = ""
                st.session_state["resume_analysis"] = ""
                update_workspace_state(
                    int(st.session_state["active_workspace_id"]),
                    summary_text="",
                    last_document_text=text,
                    provider=st.session_state["provider"],
                    model=st.session_state["model"],
                )
                _rebuild_workspace_index()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                _persist_ocr_event(st.session_state["provider"], elapsed_ms, uploaded.name, ext)
                log_ocr(f"Uploaded {safe_name} processed", category="ocr", file_type=ext, file_hash=file_hash)
                st.success(f"Extracted {len(text):,} characters from {uploaded.name}")
                status.update(label=f"Processed {uploaded.name}", state="complete")
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                _persist_ocr_event(st.session_state["provider"], elapsed_ms, uploaded.name, ext)
                log_error(f"OCR extraction failed for {safe_name}: {exc}")
                status.update(label=f"Failed {uploaded.name}", state="error")
                st.error(f"Extraction failed for {uploaded.name}: {exc}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    progress.progress(1.0)
    update_workspace_activity(int(st.session_state["active_workspace_id"]))
    _mark_activity()
    st.rerun()


def _summary_tab() -> None:
    st.markdown("### Upload or review a document")
    _upload_documents_tab()

    if st.session_state.get("uploaded_documents"):
        with st.expander("Saved documents", expanded=False):
            for document in st.session_state["uploaded_documents"]:
                cols = st.columns([4, 1, 1])
                cols[0].markdown(f"**{document.get('filename')}** · {document.get('mime_type', 'unknown')} · uploaded {document.get('upload_time')}")
                if cols[1].button("Delete", key=f"del_doc_{document['id']}"):
                    delete_document(int(document["id"]))
                    _rebuild_workspace_index()
                    st.rerun()
                if cols[2].button("Preview", key=f"preview_doc_{document['id']}"):
                    st.session_state["extracted_text"] = document.get("text_content", "")
                    st.rerun()

    if st.session_state["extracted_text"]:
        with st.expander("Extracted text preview", expanded=False):
            st.text_area("Document text", st.session_state["extracted_text"], height=240, disabled=True)

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            do_summarize = st.button("✨ Generate Summary", use_container_width=True)
        with col_info:
            st.caption(
                f"Mode: {st.session_state['summary_mode']} · Tone: {st.session_state['tone']} · Provider: {st.session_state['provider']} / {st.session_state['model']}"
            )

        if do_summarize:
            if not st.session_state["api_key"]:
                st.warning(f"Please enter your {st.session_state['provider']} API key in the sidebar.")
            elif not _check_rate_limit("summary"):
                return
            else:
                started = time.perf_counter()
                try:
                    with st.spinner(f"Generating summary with {st.session_state['provider']}…"):
                        summary = summarize_text(
                            st.session_state["extracted_text"],
                            provider=st.session_state["provider"],
                            model=st.session_state["model"],
                            api_key=st.session_state["api_key"],
                            mode=st.session_state["summary_mode"],
                            tone=st.session_state["tone"],
                        )
                    st.session_state["summary"] = summary
                    update_workspace_state(int(st.session_state["active_workspace_id"]), summary_text=summary)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    _persist_llm_event("summary", st.session_state["provider"], elapsed_ms, st.session_state["extracted_text"], summary)
                    log_provider(f"Summary generated with {st.session_state['provider']} / {st.session_state['model']}", category="provider")
                    st.success("Summary generated.")
                except GeminiAPIError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    log_error(f"Summary generation failed: {exc}")
                    st.error(f"Summary failed: {exc}")

    if st.session_state["summary"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Summary")
        st.write(st.session_state["summary"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("#### 📥 Download Summary")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("⬇️ TXT", data=to_txt(st.session_state["summary"]), file_name="eshwar_summary.txt", mime="text/plain", use_container_width=True)
        with dl2:
            st.download_button("⬇️ PDF", data=to_pdf(st.session_state["summary"]), file_name="eshwar_summary.pdf", mime="application/pdf", use_container_width=True)
        with dl3:
            st.download_button("⬇️ DOCX", data=to_docx(st.session_state["summary"]), file_name="eshwar_summary.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)


def _rag_tab() -> None:
    st.markdown("### 💬 Ask Questions About Your Workspace")
    if not st.session_state.get("uploaded_documents"):
        st.info("Upload one or more documents in the Summarize tab first.")
        return

    if st.session_state["rag_collection"] is None:
        try:
            _rebuild_workspace_index()
            st.success("Workspace indexed for retrieval.")
        except Exception as exc:
            st.error(f"Index build failed: {exc}")

    col_left, col_right = st.columns([1, 4])
    with col_left:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            clear_chat_history(int(st.session_state["active_workspace_id"]))
            st.session_state["chat_history"] = []
            st.rerun()
    with col_right:
        st.caption(f"Memory window: last {st.session_state['memory_turns']} turns · Top-k source chunks: {st.session_state['rag_top_k']}")

    _render_chat_history()
    question = st.chat_input("Ask about your uploaded documents")
    if not question or not question.strip():
        return
    if not st.session_state["api_key"]:
        st.warning("Enter your API key in the sidebar.")
        return
    if not _check_rate_limit("chat"):
        return

    user_text = question.strip()
    st.session_state["chat_history"].append({"role": "user", "text": user_text, "created_at": _utc_now()})
    _persist_chat("user", user_text)

    started = time.perf_counter()
    with st.spinner("Searching document sources and generating an answer…"):
        try:
            retrieval_context = retrieve(st.session_state["rag_collection"], user_text, k=int(st.session_state["rag_top_k"]))
            source_chunks = format_sources(retrieval_context)
            answer = answer_with_context(
                user_text,
                retrieval_context,
                provider=st.session_state["provider"],
                model=st.session_state["model"],
                api_key=st.session_state["api_key"],
                conversation_history=st.session_state["chat_history"],
                memory_turns=int(st.session_state["memory_turns"]),
            )
        except GeminiAPIError as exc:
            answer = str(exc)
            source_chunks = []
        except Exception as exc:
            answer = f"⚠️ Error: {exc}"
            source_chunks = []
            log_error(f"Chat response failed: {exc}")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    st.session_state["chat_history"].append({"role": "assistant", "text": answer, "sources": source_chunks, "created_at": _utc_now()})
    _persist_chat("assistant", answer, sources=source_chunks, provider=st.session_state["provider"], latency_ms=elapsed_ms, token_usage=_estimate_tokens(user_text, answer))
    st.rerun()


def _resume_tab() -> None:
    st.markdown("### 📋 Resume Analyzer")
    if not st.session_state["extracted_text"]:
        st.info("Upload a resume in the Summarize tab first.")
        return

    job_desc = st.text_area("Job Description (optional)", height=130, key="job_desc_input", placeholder="Paste the job description here to compare against the resume…")
    if st.button("🎯 Analyze Resume", use_container_width=False):
        if not st.session_state["api_key"]:
            st.warning("Enter your API key in the sidebar.")
        elif not _check_rate_limit("resume"):
            return
        else:
            started = time.perf_counter()
            try:
                with st.spinner("Analyzing resume…"):
                    analysis = analyze_resume_text(
                        st.session_state["extracted_text"],
                        provider=st.session_state["provider"],
                        model=st.session_state["model"],
                        api_key=st.session_state["api_key"],
                        job_desc=job_desc,
                    )
                st.session_state["resume_analysis"] = analysis
                update_workspace_state(int(st.session_state["active_workspace_id"]), summary_text=analysis)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                _persist_llm_event("resume", st.session_state["provider"], elapsed_ms, job_desc or st.session_state["extracted_text"], analysis)
                log_provider(f"Resume analysis completed with {st.session_state['provider']} / {st.session_state['model']}", category="provider")
                st.success("Resume analyzed.")
            except GeminiAPIError as exc:
                st.error(str(exc))
            except Exception as exc:
                log_error(f"Resume analysis failed: {exc}")
                st.error(f"Analysis failed: {exc}")

    if st.session_state["resume_analysis"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Resume Analysis")
        st.write(st.session_state["resume_analysis"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Download Analysis (TXT)", data=to_txt(st.session_state["resume_analysis"]), file_name="eshwar_resume_analysis.txt", mime="text/plain")


def _workspaces_tab() -> None:
    st.markdown("### 🗂️ Workspaces and File Management")
    workspace_id = st.session_state.get("active_workspace_id")
    if not workspace_id:
        st.info("No workspace selected.")
        return

    workspace = get_workspace(int(workspace_id))
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"#### {workspace['name']}")
    st.caption(f"Created {workspace['created_at']} · Updated {workspace['updated_at']} · Last opened {workspace.get('last_opened_at') or 'unknown'}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Saved documents")
    documents = list_documents(int(workspace_id))
    if not documents:
        st.info("No documents saved in this workspace yet.")
    else:
        for document in documents:
            cols = st.columns([4, 1, 1])
            cols[0].markdown(f"**{document.get('filename')}** · {document.get('mime_type', 'unknown')} · uploaded {document.get('upload_time')}")
            if cols[1].button("Delete", key=f"workspace_doc_delete_{document['id']}"):
                delete_document(int(document["id"]))
                _rebuild_workspace_index()
                st.rerun()
            if cols[2].button("Preview", key=f"workspace_doc_preview_{document['id']}"):
                st.session_state["extracted_text"] = document.get("text_content", "")
                st.rerun()

    st.markdown("#### Conversation export")
    messages = st.session_state["chat_history"]
    if messages:
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button("⬇️ TXT", data=chat_history_to_txt(messages), file_name="conversation.txt", mime="text/plain", use_container_width=True)
        with e2:
            st.download_button("⬇️ PDF", data=chat_history_to_pdf(messages), file_name="conversation.pdf", mime="application/pdf", use_container_width=True)
        with e3:
            st.download_button("⬇️ DOCX", data=chat_history_to_docx(messages), file_name="conversation.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
    else:
        st.info("No conversation to export yet.")

    st.markdown("#### Workspace management")
    st.caption("Embeddings are rebuilt from stored chunk vectors when available; duplicate chunks are filtered automatically.")


def _analytics_tab() -> None:
    st.markdown("### 📊 Analytics dashboard")
    if not st.session_state.get("current_user"):
        st.info("Sign in to view analytics.")
        return

    summary = get_analytics_summary(int(st.session_state["current_user"]["id"]), workspace_id=int(st.session_state["active_workspace_id"]))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Uploads", summary.get("uploads", 0))
    metric_cols[1].metric("OCR events", summary.get("ocr_events", 0))
    metric_cols[2].metric("Chats", summary.get("chats", 0))
    metric_cols[3].metric("Token usage", summary.get("token_usage", 0))

    st.markdown("#### Response latency")
    latency = summary.get("avg_latency", {}) or {}
    if latency:
        st.dataframe([{"event_type": key, "avg_latency_ms": round(float(value), 2) if value is not None else None} for key, value in latency.items()], use_container_width=True, hide_index=True)
    else:
        st.info("No latency data yet.")

    st.markdown("#### Provider usage")
    provider_usage = summary.get("provider_usage", []) or []
    if provider_usage:
        st.dataframe(provider_usage, use_container_width=True, hide_index=True)
    else:
        st.info("No provider events yet.")


def _health_tab() -> None:
    st.markdown("### 🩺 Health monitoring")
    report = get_health_report()
    overall = report.get("status", "degraded")
    if overall == "ok":
        st.success("All core systems healthy.")
    else:
        st.warning("One or more subsystems need attention.")
    st.json(report)


def _about_tab() -> None:
    st.markdown(
        """
        <div class="card">
        <h3>📄 Eshwar SaaS Document Platform</h3>
        <p style="color:#94a3b8;">OCR, multi-document RAG, conversational memory, saved workspaces, and analytics in a single Streamlit app.</p>
        <h4>Capabilities</h4>
        <ul>
          <li>OCR extraction from PNG, JPG, JPEG, PDF, DOCX, MP4, MOV, AVI</li>
          <li>Multi-provider summaries and chat responses</li>
          <li>Saved workspaces with persistent documents and chat history</li>
          <li>Source citations and exportable conversation history</li>
          <li>Local SQLite persistence and analytics dashboard</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workspace_overview() -> None:
    workspace = get_workspace(int(st.session_state["active_workspace_id"]))
    documents = list_documents(int(st.session_state["active_workspace_id"]))
    cols = st.columns(4)
    cols[0].metric("Documents", len(documents))
    cols[1].metric("Chat turns", len(st.session_state["chat_history"]))
    cols[2].metric("Summary", "Yes" if st.session_state.get("summary") else "No")
    cols[3].metric("Embeddings", "Loaded" if st.session_state.get("rag_collection") is not None else "None")
    st.caption(f"Workspace: {workspace['name']} · created {workspace['created_at']} · last opened {workspace.get('last_opened_at') or 'unknown'}")


def _ensure_workspace_selected() -> None:
    if st.session_state.get("authenticated") and st.session_state.get("auth_token"):
        auth_session = get_auth_session(st.session_state["auth_token"])
        if not auth_session:
            _logout()
            return
        if time.time() - float(st.session_state.get("last_active_ts", 0)) > SESSION_TIMEOUT_MINUTES * 60:
            _logout()
            return
        _mark_activity()


if not st.session_state["authenticated"]:
    _auth_screen()
    st.stop()

_ensure_workspace_selected()
if not st.session_state["authenticated"]:
    st.stop()

current_user = get_user_by_id(int(st.session_state["current_user"]["id"])) if st.session_state.get("current_user") else None
if current_user:
    st.session_state["current_user"] = current_user

_sidebar_after_auth()

st.markdown("<h1 style='text-align:center; color:#a78bfa; margin-bottom:.25rem;'>📄 Eshwar — SaaS Document Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>Authenticated workspaces · OCR · RAG · citations · analytics</p>", unsafe_allow_html=True)
st.markdown("---")

_render_workspace_overview()

tab_sum, tab_rag, tab_res, tab_ws, tab_ana, tab_health, tab_about = st.tabs(["📄 Summarize", "💬 RAG Chat", "📋 Resume Analyzer", "🗂️ Workspaces", "📊 Analytics", "🩺 Health", "ℹ️ About"])

with tab_sum:
    _summary_tab()

with tab_rag:
    _rag_tab()

with tab_res:
    _resume_tab()

with tab_ws:
    _workspaces_tab()

with tab_ana:
    _analytics_tab()

with tab_health:
    _health_tab()

with tab_about:
    _about_tab()
