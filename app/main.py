"""
main.py — Eshwar: Multimodal Document Analyzer
================================================
Tabs:
  1. 📄 Summarize   — OCR + multi-mode summary + download
  2. 💬 RAG Chat    — document Q&A with vector retrieval
  3. 📋 Resume      — resume-specific analysis
  4. ℹ️  About       — app info
"""

from __future__ import annotations

import sys
import os
import tempfile

import streamlit as st

# Ensure the app package is importable whether run from root or app/
sys.path.insert(0, os.path.dirname(__file__))

from ocr_utils import extract_text, extract_video_text, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS
from llm_utils import (
    summarize_text,
    answer_with_context,
    chunk_text,
    PROVIDER_MODELS,
    SUMMARY_MODES,
    TONES,
    GeminiAPIError,
)
from download_utils import to_txt, to_pdf, to_docx

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Eshwar — Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark UI styles ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f1117;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #161b27;
    }
    /* ── cards ── */
    .card {
        background: #1a1f2e;
        border: 1px solid #2d3448;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    /* ── success / error banners ── */
    .banner-ok {
        background: #0d3b2e;
        border-left: 4px solid #22c55e;
        padding: .6rem 1rem;
        border-radius: 6px;
        margin-bottom: .8rem;
    }
    .banner-err {
        background: #3b0d0d;
        border-left: 4px solid #ef4444;
        padding: .6rem 1rem;
        border-radius: 6px;
        margin-bottom: .8rem;
    }
    /* ── buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: .5rem 1.4rem;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .85; }
    /* ── tabs ── */
    [data-testid="stTabs"] button {
        font-weight: 600;
        color: #94a3b8;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #818cf8;
        border-bottom: 2px solid #818cf8;
    }
    /* ── inputs ── */
    textarea, input[type="text"], input[type="password"] {
        background-color: #1e2433 !important;
        color: #e0e0e0 !important;
        border-color: #2d3448 !important;
    }
    /* ── chat bubbles ── */
    .chat-user {
        background: #2d3448;
        border-radius: 10px 10px 2px 10px;
        padding: .6rem 1rem;
        margin: .4rem 0 .4rem 20%;
        font-size: .95rem;
    }
    .chat-bot {
        background: #1e2a3a;
        border-radius: 10px 10px 10px 2px;
        padding: .6rem 1rem;
        margin: .4rem 20% .4rem 0;
        font-size: .95rem;
        border-left: 3px solid #6366f1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialisation ──────────────────────────────────────────────

_DEFAULTS: dict = {
    "provider": "Gemini",
    "model": PROVIDER_MODELS["Gemini"][0],
    "api_key": "",
    "extracted_text": "",
    "summary": "",
    "summary_mode": SUMMARY_MODES[0],
    "tone": TONES[0],
    "rag_collection": None,
    "rag_chunks": [],
    "chat_history": [],        # list of {"role": "user"|"bot", "text": str}
    "last_uploaded_name": "",
    "resume_analysis": "",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ AI Provider")

    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_MODELS.keys()),
        index=list(PROVIDER_MODELS.keys()).index(st.session_state["provider"]),
        key="_sb_provider",
    )
    if provider != st.session_state["provider"]:
        st.session_state["provider"] = provider
        st.session_state["model"] = PROVIDER_MODELS[provider][0]

    model = st.selectbox(
        "Model",
        options=PROVIDER_MODELS[provider],
        index=(
            PROVIDER_MODELS[provider].index(st.session_state["model"])
            if st.session_state["model"] in PROVIDER_MODELS[provider]
            else 0
        ),
        key="_sb_model",
    )
    st.session_state["model"] = model

    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        value=st.session_state["api_key"],
        help="Never hardcoded or logged — lives only in this browser session.",
        key="_sb_apikey",
    )
    st.session_state["api_key"] = api_key

    st.markdown("---")

    st.markdown("### 📊 Summary Options")
    st.session_state["summary_mode"] = st.selectbox(
        "Mode", SUMMARY_MODES,
        index=SUMMARY_MODES.index(st.session_state["summary_mode"]),
        key="_sb_mode",
    )
    st.session_state["tone"] = st.selectbox(
        "Tone", TONES,
        index=TONES.index(st.session_state["tone"]),
        key="_sb_tone",
    )

    st.markdown("---")
    st.caption("🔒 Keys are stored only in session memory.")


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center; color:#818cf8; margin-bottom:.2rem;'>"
    "📄 Eshwar — Multimodal Document Analyzer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>"
    "OCR · Multi-mode Summaries · RAG Chat · Resume Analysis · PDF/DOCX Download"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_sum, tab_rag, tab_res, tab_about = st.tabs(
    ["📄 Summarize", "💬 RAG Chat", "📋 Resume Analyzer", "ℹ️ About"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARIZE
# ══════════════════════════════════════════════════════════════════════════════

with tab_sum:
    st.markdown("### Upload a Document")
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS | VIDEO_EXTENSIONS))
    uploaded = st.file_uploader(
        f"Supported formats: {supported}",
        type=["pdf", "docx", "jpg", "jpeg", "png", "mp4", "mov", "avi"],
        key="uploader_sum",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()

        # Size guard
        if len(file_bytes) > 15 * 1024 * 1024:
            st.markdown('<div class="banner-err">❌ File exceeds the 15 MB limit.</div>',
                        unsafe_allow_html=True)
        elif uploaded.name != st.session_state["last_uploaded_name"]:
            # New file — extract
            with st.spinner("🔍 Extracting text via OCR…"):
                tmp_path = None
                try:
                    _, ext = os.path.splitext(uploaded.name.lower())
                    if ext in VIDEO_EXTENSIONS:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".mp4") as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name
                        text = extract_video_text(tmp_path)
                    else:
                        text = extract_text(file_bytes, uploaded.name)

                    st.session_state["extracted_text"] = text
                    st.session_state["summary"] = ""
                    st.session_state["rag_collection"] = None
                    st.session_state["rag_chunks"] = []
                    st.session_state["chat_history"] = []
                    st.session_state["resume_analysis"] = ""
                    st.session_state["last_uploaded_name"] = uploaded.name

                    # Pre-build RAG index in background
                    chunks = chunk_text(text)
                    st.session_state["rag_chunks"] = chunks
                    try:
                        from rag_utils import build_index
                        st.session_state["rag_collection"] = build_index(chunks)
                    except Exception:
                        pass  # RAG index build failure is non-fatal here

                    st.markdown(
                        f'<div class="banner-ok">✅ Extracted {len(text):,} characters '
                        f'from <strong>{uploaded.name}</strong></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.markdown(
                        f'<div class="banner-err">❌ Extraction failed: {exc}</div>',
                        unsafe_allow_html=True,
                    )
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    if st.session_state["extracted_text"]:
        with st.expander("🔎 Extracted Text Preview", expanded=False):
            st.text_area(
                "Full document text",
                st.session_state["extracted_text"],
                height=220,
                disabled=True,
                key="preview_text",
            )

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            do_summarize = st.button("✨ Generate Summary", use_container_width=True)
        with col_info:
            st.caption(
                f"Mode: **{st.session_state['summary_mode']}** · "
                f"Tone: **{st.session_state['tone']}** · "
                f"Provider: **{st.session_state['provider']} / {st.session_state['model']}**"
            )

        if do_summarize:
            if not st.session_state["api_key"]:
                st.warning(f"Please enter your {provider} API key in the sidebar.")
            else:
                with st.spinner(f"🤖 Generating summary with {provider}…"):
                    try:
                        summary = summarize_text(
                            st.session_state["extracted_text"],
                            provider=st.session_state["provider"],
                            model=st.session_state["model"],
                            api_key=st.session_state["api_key"],
                            mode=st.session_state["summary_mode"],
                            tone=st.session_state["tone"],
                        )
                        st.session_state["summary"] = summary
                        st.markdown(
                            '<div class="banner-ok">✅ Summary generated successfully '
                            f'with <strong>{st.session_state["model"]}</strong>.</div>',
                            unsafe_allow_html=True,
                        )
                    except GeminiAPIError as exc:
                        # Clean, formatted Gemini-specific error — no raw traceback
                        st.markdown(
                            f'<div class="banner-err">{exc}</div>',
                            unsafe_allow_html=True,
                        )
                    except ValueError as exc:
                        st.markdown(
                            f'<div class="banner-err">⚠️ Input error: {exc}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        st.markdown(
                            f'<div class="banner-err">❌ Summary failed: {exc}</div>',
                            unsafe_allow_html=True,
                        )

    if st.session_state["summary"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Summary")
        st.write(st.session_state["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### 📥 Download Summary")
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            st.download_button(
                "⬇️ TXT",
                data=to_txt(st.session_state["summary"]),
                file_name="eshwar_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl2:
            try:
                pdf_bytes = to_pdf(st.session_state["summary"])
                st.download_button(
                    "⬇️ PDF",
                    data=pdf_bytes,
                    file_name="eshwar_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"PDF unavailable: {e}")
        with dl3:
            try:
                docx_bytes = to_docx(st.session_state["summary"])
                st.download_button(
                    "⬇️ DOCX",
                    data=docx_bytes,
                    file_name="eshwar_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"DOCX unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RAG CHAT
# ══════════════════════════════════════════════════════════════════════════════

with tab_rag:
    st.markdown("### 💬 Ask Questions About Your Document")

    if not st.session_state["extracted_text"]:
        st.info("📂 Upload a document in the **Summarize** tab first.")
    else:
        # Ensure RAG index exists
        if st.session_state["rag_collection"] is None and st.session_state["rag_chunks"]:
            with st.spinner("Building document index…"):
                try:
                    from rag_utils import build_index
                    st.session_state["rag_collection"] = build_index(
                        st.session_state["rag_chunks"]
                    )
                    st.markdown(
                        '<div class="banner-ok">✅ Document indexed for Q&A.</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.markdown(
                        f'<div class="banner-err">❌ Index build failed: {exc}</div>',
                        unsafe_allow_html=True,
                    )

        # Render chat history
        for msg in st.session_state["chat_history"]:
            css = "chat-user" if msg["role"] == "user" else "chat-bot"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="{css}">{icon} {msg["text"]}</div>',
                unsafe_allow_html=True,
            )

        # Input
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the key findings?",
            key="rag_question_input",
        )
        c1, c2 = st.columns([1, 4])
        with c1:
            ask_btn = st.button("🔍 Ask", use_container_width=True)
        with c2:
            if st.button("🗑️ Clear Chat", use_container_width=False):
                st.session_state["chat_history"] = []
                st.rerun()

        if ask_btn and question.strip():
            if not st.session_state["api_key"]:
                st.warning("Enter your API key in the sidebar.")
            else:
                st.session_state["chat_history"].append(
                    {"role": "user", "text": question}
                )
                with st.spinner("Searching document…"):
                    try:
                        from rag_utils import retrieve
                        context = retrieve(
                            st.session_state["rag_collection"], question
                        )
                        answer = answer_with_context(
                            question,
                            context,
                            provider=st.session_state["provider"],
                            model=st.session_state["model"],
                            api_key=st.session_state["api_key"],
                        )
                    except GeminiAPIError as exc:
                        answer = str(exc)   # already a clean user-facing string
                    except Exception as exc:
                        answer = f"⚠️ Error: {exc}"

                st.session_state["chat_history"].append(
                    {"role": "bot", "text": answer}
                )
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESUME ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

with tab_res:
    st.markdown("### 📋 Resume Analyzer")
    st.caption("Upload a resume PDF/DOCX in the Summarize tab, then analyze it here.")

    if not st.session_state["extracted_text"]:
        st.info("📂 Upload a resume in the **Summarize** tab first.")
    else:
        job_desc = st.text_area(
            "Job Description (optional — paste to get match analysis)",
            height=130,
            key="job_desc_input",
            placeholder="Paste the job description here to compare against the resume…",
        )

        if st.button("🎯 Analyze Resume", use_container_width=False):
            if not st.session_state["api_key"]:
                st.warning("Enter your API key in the sidebar.")
            else:
                resume_text = st.session_state["extracted_text"]
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

                with st.spinner("🤖 Analyzing resume…"):
                    try:
                        analysis = answer_with_context(
                            base_prompt,
                            [resume_text[:8000]],
                            provider=st.session_state["provider"],
                            model=st.session_state["model"],
                            api_key=st.session_state["api_key"],
                        )
                        st.session_state["resume_analysis"] = analysis
                        st.markdown(
                            '<div class="banner-ok">✅ Resume analyzed.</div>',
                            unsafe_allow_html=True,
                        )
                    except GeminiAPIError as exc:
                        st.markdown(
                            f'<div class="banner-err">{exc}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        st.markdown(
                            f'<div class="banner-err">❌ Analysis failed: {exc}</div>',
                            unsafe_allow_html=True,
                        )

        if st.session_state["resume_analysis"]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Resume Analysis")
            st.write(st.session_state["resume_analysis"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Analysis (TXT)",
                data=to_txt(st.session_state["resume_analysis"]),
                file_name="eshwar_resume_analysis.txt",
                mime="text/plain",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.markdown(
        """
        <div class="card">
        <h3>📄 Eshwar — Multimodal Document Analyzer</h3>
        <p style="color:#94a3b8;">Production-grade OCR + LLM pipeline.</p>

        <h4>Features</h4>
        <ul>
          <li>OCR extraction from PNG, JPG, JPEG, PDF, DOCX</li>
          <li>Multi-mode summaries (Concise / Detailed / Bullets / Executive / Technical)</li>
          <li>Tone control (Neutral / Formal / Casual / Academic)</li>
          <li>RAG-powered document Q&A (ChromaDB + sentence-transformers)</li>
          <li>Resume analysis with optional job-description matching</li>
          <li>Download as TXT / PDF / DOCX</li>
        </ul>

        <h4>Providers</h4>
        <ul>
          <li><strong>Gemini</strong> — gemini-1.5-flash-latest, gemini-1.5-pro-latest</li>
          <li><strong>OpenAI</strong> — gpt-4o-mini, gpt-4o, gpt-3.5-turbo</li>
          <li><strong>Groq</strong> — llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768</li>
          <li><strong>Claude</strong> — claude-3-5-haiku-20241022, claude-3-5-sonnet-20241022</li>
        </ul>

        <h4>Stability Notes</h4>
        <ul>
          <li>Groq uses OpenAI-compatible base_url with automatic model fallback</li>
          <li>All providers use tenacity retry (3 attempts, exponential back-off)</li>
          <li>OCR images are preprocessed (greyscale → sharpen → contrast)</li>
          <li>Temp files always cleaned up; corrupt images caught early</li>
          <li>Session state fully initialized — no rerun loops</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
