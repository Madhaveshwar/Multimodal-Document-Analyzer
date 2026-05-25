# Eshwar — Multimodal Document Analyzer

Production-grade OCR + LLM pipeline built with Streamlit.

---

## Features

| Feature | Details |
|---|---|
| OCR extraction | PNG, JPG, JPEG, PDF (native + OCR fallback), DOCX |
| OCR preprocessing | Greyscale → sharpen → contrast boost |
| Summary modes | Concise, Detailed, Bullet Points, Executive, Technical |
| Tone control | Neutral, Formal, Casual, Academic |
| Provider support | Gemini, OpenAI, Groq, Claude |
| Groq models | llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768 |
| RAG Q&A | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| Resume analysis | Skills, strengths, ATS score, job-description matching |
| Downloads | TXT, PDF (ReportLab), DOCX (python-docx) |
| Retry logic | tenacity — 3 attempts, exponential back-off |
| Groq fallback | Auto-tries next allowed model on failure |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **System requirement:** Tesseract OCR must be installed.
>
> - **Ubuntu/Debian:** `sudo apt install tesseract-ocr poppler-utils`
> - **macOS:** `brew install tesseract poppler`
> - **Windows:** Install from https://github.com/UB-Mannheim/tesseract/wiki

### 2. Run the app

```bash
streamlit run app/main.py
```

Open http://localhost:8501

### 3. Docker

```bash
docker build -t eshwar .
docker run -p 8501:8501 eshwar
```

---

## API Keys

Enter your key in the sidebar. Keys are **never** hardcoded or logged.

| Provider | Environment variable |
|---|---|
| Gemini | `GOOGLE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |

You can also pass keys via environment variables so the sidebar auto-fills:

```bash
export GROQ_API_KEY=gsk_...
streamlit run app/main.py
```

---

## Changelog (stabilization pass)

### Fixed
- **Groq deprecated models removed** — replaced with `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`
- **Groq automatic fallback** — if primary model fails, tries next allowed model before raising
- **OpenAI SDK** — `from openai import OpenAI` throughout; no deprecated `openai.ChatCompletion`
- **Groq SDK** — uses OpenAI-compatible `base_url`; no `groq` SDK import needed
- **Gemini** — `google-generativeai` direct; removed all LangChain wrappers
- **Requirements** — removed LangChain, pydantic v1 pin, conflicting chromadb pins; upgraded to stable modern versions
- **OCR pipeline** — added `img.verify()` for early corrupt-image detection; temp files always cleaned via `finally`
- **OCR preprocessing** — greyscale + sharpen + contrast applied before Tesseract
- **RAG index** — ephemeral ChromaDB client; cosine similarity; safe retrieval (returns `[]` on error)
- **Session state** — all keys initialized in `_DEFAULTS`; provider switching resets model to valid default; no rerun loops
- **Download** — TXT/PDF/DOCX all produce valid, non-corrupt files; PDF uses ReportLab
- **Retry** — `tenacity` applied to all provider calls (3 attempts, exp back-off)
- **UI** — dark theme preserved; banner-ok/banner-err for feedback; chat bubbles styled

### Added
- `rag_utils.py` — dedicated RAG module (build_index, retrieve)
- `download_utils.py` — dedicated download module (to_txt, to_pdf, to_docx)
- Resume analyzer tab with job-description matching
- About tab with full feature/provider listing

### Not changed
- Dark UI theme
- Tab layout
- Core OCR logic
- Provider API call structure
