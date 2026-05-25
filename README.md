# Multimodal Document Analyzer

A lightweight Streamlit app for OCR, summarization, source-aware RAG chat, and resume analysis.

## Highlights

- Upload images, PDFs, DOCX files, and video files
- Extract text with OCR and summarize the results
- Ask grounded questions over your uploaded documents with ChromaDB and sentence-transformers
- Analyze resumes against a job description
- Export summaries and answers as TXT, PDF, or DOCX
- Optional LangSmith tracing for LLM and retrieval observability

## Quick Start

### Local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

### Docker

```bash
docker build -t multimodal-analyzer .
docker run -p 8501:8501 multimodal-analyzer
```

## Required system dependency

Install Tesseract OCR on the host or container:

- Ubuntu/Debian: `sudo apt install tesseract-ocr poppler-utils`
- macOS: `brew install tesseract poppler`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

## Environment variables

Set the provider keys you plan to use in `.env` or your deployment secrets:

- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `ANTHROPIC_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_TRACING`

Optional local-path overrides:

- `APP_LOG_DIR`
- `APP_DATA_DIR`
- `APP_DB_PATH`

## Project layout

- `app/main.py` — Streamlit UI and app flow
- `app/ocr_utils.py` — OCR and video frame extraction
- `app/rag_utils.py` — chunking, embedding, and retrieval
- `app/llm_utils.py` — provider wrappers and trace helpers
- `app/download_utils.py` — export helpers

## Deployment notes

- Streamlit Cloud: set your secrets in the app settings and keep `app/main.py` as the entry point.
- Docker: the container image runs on port `8501`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
