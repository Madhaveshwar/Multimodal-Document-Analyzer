# Architecture

This document describes the production architecture of the Multimodal AI SaaS platform, including the request path, OCR/video OCR processing, provider abstraction, RAG retrieval, persistence, deployment, observability, and operational safeguards.

## System Architecture

```mermaid
flowchart TB
    U[User] --> B[Browser]
    B --> S[Streamlit Frontend]

    S --> A[Authentication Layer]
    S --> W[Workspace & Session Manager]
    S --> O[OCR / Video OCR Pipeline]
    S --> R[RAG Retrieval Layer]
    S --> E[Export System]
    S --> H[Health UI]

    A --> DB[(SQLite Persistence)]
    W --> DB
    O --> DB
    R --> DB
    E --> DB

    O --> V[(ChromaDB Vector Store)]
    R --> V

    S --> P[Provider Abstraction Layer]
    P --> G[Gemini]
    P --> OAI[OpenAI]
    P --> GR[Groq]
    P --> C[Claude]

    P --> LS[LangSmith Tracing]
    O --> LS
    R --> LS

    H --> HC[FastAPI Health Service]
    HC --> DB
    HC --> V
    HC --> OCRH[OCR Check]

    subgraph Deployment
        D1[Docker Container: Streamlit App]
        D2[Docker Container: Health API]
        VOLUME1[(Shared Data Volume)]
        VOLUME2[(Shared Log Volume)]
    end

    S -. runs in .-> D1
    HC -. runs in .-> D2
    DB -. persists to .-> VOLUME1
    LS -. logs to .-> VOLUME2
```

## Component Flow

```mermaid
flowchart LR
    U[User Action] --> F[Upload / Chat / Resume / Workspace]
    F --> S[Streamlit UI Handler]
    S --> V1{Authenticated?}
    V1 -- No --> L[Login / Signup]
    V1 -- Yes --> WS[Resolve Active Workspace]
    WS --> P1{Action Type}

    P1 -->|Upload| OCR[OCR / Video OCR]
    P1 -->|Chat| RAG[RAG Retrieval]
    P1 -->|Resume| RES[Resume Analysis]
    P1 -->|Export| EXP[Export Generator]

    OCR --> DB[(SQLite)]
    OCR --> CH[(ChromaDB)]
    RAG --> CH
    RAG --> LLM[Provider Abstraction]
    RES --> LLM
    EXP --> DB
    LLM --> LS[LangSmith]

    OCR --> UI[Rendered Results]
    RAG --> UI
    RES --> UI
    EXP --> UI
    UI --> DB
```

## OCR Pipeline

```mermaid
flowchart TB
    I[Uploaded File] --> T{File Type}
    T -->|Image| IMG[Image OCR]
    T -->|PDF| PDF[PDF Native Text Extraction]
    T -->|PDF OCR fallback| OCRPDF[PDF Page OCR]
    T -->|DOCX| DOCX[DOCX Text Extraction]
    T -->|Video| VID[Video OCR]

    IMG --> PRE[Preprocess Image]
    OCRPDF --> PRE
    VID --> FR[Frame Sampling]
    FR --> PREV[Preprocess Video Frame]

    PRE --> TR[Tesseract OCR]
    PREV --> TR
    PDF --> NAT[Native Text]
    NAT --> MIX[Merge / Normalize Text]
    TR --> CONF{Low Confidence or Sparse Text?}
    CONF -- Yes --> RETRY[Retry with stronger preprocessing]
    CONF -- No --> MIX
    RETRY --> MIX

    MIX --> SAVE[Persist Document + Chunks]
    SAVE --> SQLITE[(SQLite)]
    SAVE --> CHROMA[(ChromaDB)]
```

## RAG Workflow

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant DB as SQLite
    participant VS as ChromaDB
    participant LLM as Provider Layer
    participant LS as LangSmith

    User->>UI: Ask a question
    UI->>DB: Load workspace, chat history, and document chunks
    UI->>VS: Retrieve semantically relevant chunks
    VS-->>UI: Ranked chunks with metadata
    UI->>UI: Build grounded prompt with citations
    UI->>LLM: Send prompt and model settings
    LLM-->>UI: Generated answer
    UI->>DB: Save chat turn and sources
    UI->>LS: Trace request, retrieval, and provider call
    UI-->>User: Answer with citations
```

## Deployment Architecture

```mermaid
flowchart TB
    Browser[Browser] --> App[Streamlit App Container]
    Browser --> Health[FastAPI Health Container]

    App --> Data[(Shared Data Volume)]
    App --> Logs[(Shared Log Volume)]
    App --> Ext[External LLM APIs]
    App --> LS[LangSmith]

    Health --> Data
    Health --> Logs

    Compose[Docker Compose] --> App
    Compose --> Health
    Compose --> Env[.env / Environment Variables]
    Env --> App
    Env --> Health
```

## Request Flow

1. The user opens the Streamlit interface and authenticates.
2. The app resolves the active workspace and loads stored documents, chat history, and analytics.
3. Uploads go through file classification, OCR or native extraction, persistence, and chunking.
4. Chat questions trigger retrieval against the vector store and the persisted chunk metadata.
5. The prompt builder combines the latest sources with recent conversation memory.
6. The provider abstraction layer sends the request to the selected model.
7. The response is rendered in the UI, traced in LangSmith, and persisted to SQLite.
8. Export actions generate TXT, PDF, or DOCX artifacts from the saved content.

## Deployment Explanation

The platform runs as two containers in Docker Compose:

1. The Streamlit container hosts the product UI, OCR pipeline, RAG orchestration, and persistence writes.
2. The FastAPI container exposes a lightweight `/health` endpoint for probes and operational checks.
3. Both services share the same data and log volumes so database files, exported artifacts, and rotating logs remain durable across restarts.
4. Configuration comes from `.env` and runtime environment variables, which keeps secrets out of the repository and supports promotion between environments.

The Docker image bundles the runtime Python dependencies and the system packages required by Tesseract and PDF conversion. This keeps the deployment self-contained and avoids relying on host-installed OCR tooling.

## Security Notes

1. Authentication is required before workspace access, document persistence, or chat history retrieval.
2. Passwords are hashed, and session tokens are stored server-side rather than embedded in the UI.
3. Uploaded filenames are sanitized before they are stored or displayed.
4. Rate limiting is applied to high-cost actions such as uploads, summaries, chat generation, and resume analysis.
5. Secrets are loaded from environment variables and masked in logs through the centralized logging filter.
6. The health endpoint exposes only operational status and no user content.
7. The production container uses a minimal Python base image and avoids shipping unnecessary build artifacts.

## Observability

The observability model has three layers:

1. Rotating file logs capture requests, OCR operations, provider calls, and errors in separate log channels.
2. LangSmith traces the OCR, retrieval, and model-generation path for debugging, latency analysis, and prompt inspection.
3. The FastAPI health service reports database, OCR, and vector-store readiness for uptime monitoring and deployment validation.

Together, these layers cover runtime diagnostics, cost-sensitive AI operations, and environment health.

## Caching Strategy

1. The embedding model and Chroma client are cached at process scope to avoid repeated initialization overhead.
2. OCR preprocessing is applied only when a file or frame is actually processed, and extracted results are persisted so the same content does not need to be re-OCR'd on every query.
3. Session state keeps the active workspace, chat context, and UI controls in memory for fast interaction during a browser session.
4. Retrieval is limited to the top-ranked chunks so prompt size stays bounded and model calls remain efficient.

## Persistence Strategy

1. SQLite stores users, auth sessions, workspaces, documents, document chunks, chat history, and analytics events.
2. ChromaDB stores semantic vectors for document chunks so retrieval stays fast and local.
3. Shared Docker volumes preserve the database and logs across container restarts.
4. The app writes derived outputs such as exports and OCR results only after successful processing, which keeps persisted state consistent with the visible UI.
5. The database layer uses WAL mode, a busy timeout, and retryable writes to reduce lock contention during multi-action use.

## Summary

The platform is designed as a production SaaS document assistant with a clear separation between the UI, the AI services, persistence, and operations. Streamlit handles the user experience, SQLite and ChromaDB provide local durable storage, LangSmith provides traceability, and Docker Compose packages the app with a health surface suitable for deployment.