# Folder Structure

This guide explains the repository layout for contributors, reviewers, and hiring managers.

```text
Multimodal-Document-Analyzer-main/
├── app/
│   ├── main.py              # Streamlit entrypoint and UI orchestration
│   ├── database.py          # SQLite persistence and workspace storage
│   ├── ocr_utils.py         # Image, PDF, and video OCR pipeline
│   ├── rag_utils.py         # Chunking, embeddings, retrieval, and source formatting
│   ├── llm_utils.py         # Provider abstraction for Gemini, OpenAI, Groq, and Claude
│   ├── download_utils.py    # TXT, PDF, and DOCX export helpers
│   ├── health_api.py        # FastAPI health service
│   ├── health_utils.py      # Health checks for database, OCR, and vector store
│   ├── config.py            # Centralized runtime configuration
│   ├── logging_utils.py     # Rotating logs and secret masking
│   ├── security_utils.py    # Password and session helpers
│   └── tracing_utils.py     # LangSmith tracing helper
├── ARCHITECTURE.md          # System and deployment architecture diagrams
├── README.md                # Public portfolio overview and setup guide
├── ASSETS.md                # Banner, screenshot, and demo guidance
├── CONTRIBUTING.md          # Contribution workflow
├── CODE_OF_CONDUCT.md       # Community guidelines
├── SECURITY.md              # Responsible disclosure guidance
├── CHANGELOG.md             # Release history
├── LICENSE                  # MIT license
├── requirements.txt         # Python dependencies
├── Dockerfile               # Streamlit app container image
├── docker-compose.yml       # Local two-service deployment
├── .env.example             # Safe environment template
└── .gitignore               # Local-only and generated artifact exclusions
```

## What Matters Most

- `app/main.py` coordinates the production workflow and user-facing experience.
- `app/database.py` is the persistence backbone for auth, workspaces, documents, chats, and analytics.
- `app/ocr_utils.py` and `app/rag_utils.py` are the core intelligence layers.
- `app/health_api.py` provides an operational health endpoint for deployments.
- `ARCHITECTURE.md` should be the first stop for anyone trying to understand the system design.