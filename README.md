# Eshwar — Multimodal Document Analyzer

![Banner](https://placehold.co/1600x500/0f172a/e2e8f0?text=Multimodal+AI+SaaS+Platform)

Production-grade multimodal AI SaaS for OCR, video text extraction, source-aware RAG, document intelligence, and workspace-based collaboration.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-111827)](https://smith.langchain.com/)

## Hero

This platform turns unstructured documents, images, PDFs, and videos into searchable, cited, production-ready answers. It combines Streamlit, OCR, retrieval, persistence, tracing, and containerized deployment into one portfolio-ready SaaS experience.

## Why This Project Matters

- It reflects real enterprise AI workflows where documents arrive in many formats and need structured intelligence.
- It demonstrates multimodal ingestion across images, PDFs, DOCX files, and video frames.
- It uses a production RAG architecture with source-aware responses instead of generic chat output.
- It includes observability with rotating logs, request categories, and LangSmith traces.
- It shows how to package AI into a deployable product rather than a notebook prototype.

## Architecture Preview

```mermaid
flowchart LR
	User --> UI[Streamlit Frontend]
	UI --> Auth[Authentication]
	UI --> OCR[OCR / Video OCR]
	UI --> RAG[RAG Retrieval]
	UI --> Export[Export System]
	OCR --> DB[(SQLite)]
	OCR --> VS[(ChromaDB)]
	RAG --> VS
	RAG --> LLM[Provider Abstraction]
	LLM --> Models[Gemini / OpenAI / Groq / Claude]
	UI --> Trace[LangSmith Tracing]
	UI --> Health[FastAPI Health Service]
	Health --> DB
	Health --> VS
```

For full diagrams and deployment topology, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Feature Highlights

- Multimodal upload handling for images, PDFs, DOCX, and video files
- OCR preprocessing tuned for better extraction quality and fallback behavior
- Source-aware RAG responses with citations and conversation memory
- Persistent workspaces with saved chats, documents, chunks, and analytics
- Provider abstraction for Gemini, OpenAI, Groq, and Claude
- Resume analysis with job-description matching and ATS-style guidance
- Export generation in TXT, PDF, and DOCX formats
- Production health checks, rotating logs, and tracing support

## Screenshots

Add these public-release assets when you are ready:

| Placeholder | Recommended use |
|---|---|
| `docs/images/hero-banner.png` | Top-of-README banner |
| `docs/images/dashboard-overview.png` | Main product overview |
| `docs/images/ocr-upload-flow.png` | OCR and upload experience |
| `docs/images/rag-chat.png` | Source-aware Q&A flow |
| `docs/images/analytics.png` | Operational and usage analytics |

See [ASSETS.md](ASSETS.md) for banner and demo guidance.

## Demo

Suggested public demo assets:

| Asset | Purpose |
|---|---|
| `docs/demo/workflow-demo.gif` | Upload → OCR → retrieval → answer |
| `docs/demo/rag-answer-demo.gif` | Grounded answer with citations |
| `docs/demo/deployment-demo.gif` | Start-up and health check flow |

If you do not have media yet, keep this section in place and add the GIFs later. It signals portfolio readiness and helps recruiters understand the product quickly.

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | Streamlit |
| OCR | pytesseract, Pillow, pdfplumber, pdf2image, OpenCV |
| LLM Providers | Gemini, OpenAI, Groq, Claude |
| Retrieval | ChromaDB, sentence-transformers |
| Persistence | SQLite |
| Observability | LangSmith, rotating logs |
| Deployment | Docker, Docker Compose, FastAPI health service |
| Utilities | python-dotenv, tenacity, reportlab, python-docx |

## Production Features

- Environment-based configuration with safe local defaults
- Secret masking in logs
- Rate limiting for expensive user actions
- Durable workspace persistence for auth and chat history
- Health endpoint for deployment probes and monitoring
- Dockerized runtime for repeatable builds and local parity
- Sanitized upload handling and production-focused file management

## Observability

The app includes three observability layers:

- Rotating logs for requests, OCR actions, provider calls, and errors
- LangSmith traces for prompt, retrieval, and model-call visibility
- A FastAPI health service for database, OCR, and vector-store checks

## Security

- API keys are stored in environment variables, not hardcoded
- Uploaded filenames are sanitized before reuse
- Authentication gates access to workspaces and stored content
- The health endpoint exposes only operational status
- The repository ignores `.env`, local databases, caches, and log files

## Roadmap

- Add richer screenshot and demo media for the public portfolio release
- Introduce stronger analytics and usage insights
- Expand deployment guides for additional cloud targets
- Add test coverage for public release workflows
- Extend persistence abstractions for future database portability

## Folder Structure

See [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) for a documented repository layout and module responsibilities.

## Deployment Instructions

### Docker Local Deployment

```bash
cp .env.example .env
docker compose up --build
```

The app runs on `http://localhost:8501` and the health service runs on `http://localhost:8502/health`.

### Streamlit Cloud

1. Push the repository to GitHub.
2. Set the app entry point to `app/main.py`.
3. Add environment variables in the Streamlit Cloud settings page.
4. Confirm that the Tesseract binary is available in the target environment or adapt OCR settings accordingly.

### Render

1. Create a new Web Service from the GitHub repository.
2. Use `streamlit run app/main.py --server.address=0.0.0.0 --server.port=$PORT` as the start command.
3. Configure the environment variables from `.env.example`.
4. If deploying the health service separately, run `uvicorn app.health_api:app --host 0.0.0.0 --port 8502` as a second service.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

The system requires Tesseract OCR on the host or container image.

## API Keys

| Provider | Environment variable |
|---|---|
| Gemini | `GOOGLE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
