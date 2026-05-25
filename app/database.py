"""SQLite persistence layer for authentication, workspaces, documents, chats, and analytics."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from config import get_settings
from logging_utils import get_logger
from security_utils import generate_session_token, hash_password, verify_password

logger = get_logger("app.database")

SETTINGS = get_settings()
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SETTINGS.data_dir
DB_PATH = SETTINGS.db_path
SESSION_TIMEOUT_HOURS = 8
DB_BUSY_TIMEOUT_MS = 30_000
DB_RETRY_ATTEMPTS = 3
DB_RETRY_BACKOFF_SECONDS = 0.2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


class DatabaseBackend(ABC):
    """Backend contract for future database migrations."""

    name: str = "base"

    @abstractmethod
    def connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        raise NotImplementedError


class SQLiteBackend(DatabaseBackend):
    name = "sqlite"

    def connect(self) -> sqlite3.Connection:
        _ensure_data_dir()
        conn = sqlite3.connect(
            DB_PATH,
            check_same_thread=False,
            timeout=DB_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {DB_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def health_check(self) -> dict[str, Any]:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1").fetchone()
                row = conn.execute("PRAGMA quick_check").fetchone()
                return {"ok": True, "backend": self.name, "db_path": str(DB_PATH), "quick_check": row[0] if row else "ok"}
        except Exception as exc:
            return {"ok": False, "backend": self.name, "db_path": str(DB_PATH), "error": str(exc)}


DATABASE_BACKEND: DatabaseBackend = SQLiteBackend()


def _connect() -> sqlite3.Connection:
    return DATABASE_BACKEND.connect()


def _is_retryable_sqlite_error(exc: Exception) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return any(term in message for term in ("database is locked", "database is busy", "locked", "busy"))


def retry_sqlite(fn):
    def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(DB_RETRY_ATTEMPTS):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if not _is_retryable_sqlite_error(exc) or attempt == DB_RETRY_ATTEMPTS - 1:
                    raise
                last_exc = exc
                time.sleep(DB_RETRY_BACKOFF_SECONDS * (attempt + 1))
        if last_exc:
            raise last_exc
    return wrapper


@contextmanager
def db_connection() -> Iterable[sqlite3.Connection]:
    conn = _connect()
    try:
        conn.execute("BEGIN")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def db_health_check() -> tuple[bool, dict[str, Any]]:
    result = DATABASE_BACKEND.health_check()
    return bool(result.get("ok")), result


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_opened_at TEXT,
    summary_text TEXT,
    last_document_text TEXT,
    provider TEXT,
    model TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    upload_time TEXT NOT NULL,
    mime_type TEXT,
    size_bytes INTEGER NOT NULL,
    source_type TEXT,
    text_content TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    UNIQUE(workspace_id, file_hash)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    workspace_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY(workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    UNIQUE(workspace_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    sources_json TEXT,
    provider TEXT,
    latency_ms REAL,
    token_usage INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analytics_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    workspace_id INTEGER,
    event_type TEXT NOT NULL,
    provider TEXT,
    token_usage INTEGER,
    latency_ms REAL,
    details_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);
"""


@retry_sqlite
def init_db() -> None:
    with db_connection() as conn:
        conn.executescript(SCHEMA)
        _ensure_workspace_columns(conn)


def _ensure_workspace_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(workspaces)").fetchall()}
    additions = {
        "summary_text": "ALTER TABLE workspaces ADD COLUMN summary_text TEXT",
        "last_document_text": "ALTER TABLE workspaces ADD COLUMN last_document_text TEXT",
        "provider": "ALTER TABLE workspaces ADD COLUMN provider TEXT",
        "model": "ALTER TABLE workspaces ADD COLUMN model TEXT",
    }
    for column_name, statement in additions.items():
        if column_name not in existing:
            conn.execute(statement)


@retry_sqlite
def create_user(username: str, email: str, password: str) -> dict[str, Any]:
    if not username.strip() or not email.strip() or not password:
        raise ValueError("Username, email, and password are required.")

    created_at = _utc_now()
    password_hash = hash_password(password)
    with db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), password_hash, created_at),
        )
        user_id = int(cur.lastrowid)
    return get_user_by_id(user_id)


def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_identifier(identifier: str) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE lower(username) = lower(?) OR lower(email) = lower(?)",
            (identifier.strip(), identifier.strip()),
        ).fetchone()
    return dict(row) if row else None


@retry_sqlite
def authenticate_user(identifier: str, password: str) -> Optional[dict[str, Any]]:
    user = get_user_by_identifier(identifier)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    with db_connection() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (_utc_now(), user["id"]),
        )
    return get_user_by_id(user["id"])


@retry_sqlite
def create_auth_session(user_id: int) -> dict[str, Any]:
    token = generate_session_token()
    created_at = _utc_now()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=SESSION_TIMEOUT_HOURS)).isoformat(timespec="seconds")
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO auth_sessions (token, user_id, created_at, last_active_at, expires_at, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            (token, user_id, created_at, created_at, expires_at),
        )
    return {"token": token, "created_at": created_at, "expires_at": expires_at}


def get_auth_session(token: str) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM auth_sessions WHERE token = ? AND is_active = 1",
            (token,),
        ).fetchone()
    if not row:
        return None
    session = dict(row)
    if session["expires_at"] <= _utc_now():
        end_auth_session(token)
        return None
    return session


@retry_sqlite
def touch_auth_session(token: str) -> None:
    with db_connection() as conn:
        conn.execute(
            "UPDATE auth_sessions SET last_active_at = ?, expires_at = ? WHERE token = ? AND is_active = 1",
            (
                _utc_now(),
                (datetime.now(timezone.utc) + timedelta(hours=SESSION_TIMEOUT_HOURS)).isoformat(timespec="seconds"),
                token,
            ),
        )


@retry_sqlite
def end_auth_session(token: str) -> None:
    with db_connection() as conn:
        conn.execute("UPDATE auth_sessions SET is_active = 0 WHERE token = ?", (token,))


@retry_sqlite
def create_workspace(user_id: int, name: str) -> dict[str, Any]:
    workspace_name = name.strip() or "Workspace"
    now = _utc_now()
    with db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO workspaces (user_id, name, created_at, updated_at, last_opened_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, workspace_name, now, now, now),
        )
        workspace_id = int(cur.lastrowid)
    return get_workspace(workspace_id)


def list_workspaces(user_id: int) -> list[dict[str, Any]]:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM workspaces WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_workspace(workspace_id: int) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
    return dict(row) if row else None


def get_workspace_by_name(user_id: int, name: str) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM workspaces WHERE user_id = ? AND lower(name) = lower(?)",
            (user_id, name.strip()),
        ).fetchone()
    return dict(row) if row else None


@retry_sqlite
def rename_workspace(workspace_id: int, new_name: str) -> dict[str, Any]:
    if not new_name.strip():
        raise ValueError("Workspace name cannot be empty.")
    with db_connection() as conn:
        conn.execute(
            "UPDATE workspaces SET name = ?, updated_at = ? WHERE id = ?",
            (new_name.strip(), _utc_now(), workspace_id),
        )
    return get_workspace(workspace_id)


@retry_sqlite
def delete_workspace(workspace_id: int) -> None:
    with db_connection() as conn:
        conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))


@retry_sqlite
def update_workspace_activity(workspace_id: int) -> None:
    with db_connection() as conn:
        conn.execute(
            "UPDATE workspaces SET updated_at = ?, last_opened_at = ? WHERE id = ?",
            (_utc_now(), _utc_now(), workspace_id),
        )


@retry_sqlite
def update_workspace_state(
    workspace_id: int,
    *,
    summary_text: str | None = None,
    last_document_text: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    with db_connection() as conn:
        workspace = get_workspace(workspace_id)
        if not workspace:
            return
        conn.execute(
            """
            UPDATE workspaces
            SET summary_text = COALESCE(?, summary_text),
                last_document_text = COALESCE(?, last_document_text),
                provider = COALESCE(?, provider),
                model = COALESCE(?, model),
                updated_at = ?
            WHERE id = ?
            """,
            (summary_text, last_document_text, provider, model, _utc_now(), workspace_id),
        )


@retry_sqlite
def save_document(
    workspace_id: int,
    *,
    filename: str,
    file_hash: str,
    upload_time: str,
    mime_type: str,
    size_bytes: int,
    source_type: str,
    text_content: str,
    metadata: dict[str, Any],
) -> Optional[dict[str, Any]]:
    created_at = _utc_now()
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    with db_connection() as conn:
        existing = conn.execute(
            "SELECT * FROM documents WHERE workspace_id = ? AND file_hash = ?",
            (workspace_id, file_hash),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE documents SET filename = ?, upload_time = ?, mime_type = ?, size_bytes = ?, source_type = ?, text_content = ?, metadata_json = ?, created_at = ? WHERE id = ?",
                (filename, upload_time, mime_type, size_bytes, source_type, text_content, metadata_json, created_at, existing["id"]),
            )
            document_id = int(existing["id"])
        else:
            cur = conn.execute(
                """
                INSERT INTO documents (workspace_id, filename, file_hash, upload_time, mime_type, size_bytes, source_type, text_content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (workspace_id, filename, file_hash, upload_time, mime_type, size_bytes, source_type, text_content, metadata_json, created_at),
            )
            document_id = int(cur.lastrowid)
    return get_document(document_id)


def get_document(document_id: int) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if not row:
        return None
    document = dict(row)
    document["metadata"] = json.loads(document.get("metadata_json") or "{}")
    return document


def list_documents(workspace_id: int) -> list[dict[str, Any]]:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE workspace_id = ? ORDER BY created_at DESC, id DESC",
            (workspace_id,),
        ).fetchall()
    documents: list[dict[str, Any]] = []
    for row in rows:
        document = dict(row)
        document["metadata"] = json.loads(document.get("metadata_json") or "{}")
        documents.append(document)
    return documents


@retry_sqlite
def delete_document(document_id: int) -> None:
    with db_connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))


@retry_sqlite
def clear_workspace_embeddings(workspace_id: int) -> None:
    with db_connection() as conn:
        conn.execute("DELETE FROM document_chunks WHERE workspace_id = ?", (workspace_id,))


@retry_sqlite
def save_document_chunks(document_id: int, workspace_id: int, chunk_records: list[dict[str, Any]]) -> None:
    if not chunk_records:
        return
    with db_connection() as conn:
        for record in chunk_records:
            metadata = dict(record)
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            chunk_id = str(record.get("chunk_id") or "")
            embedding = record.get("embedding")
            if embedding is None and record.get("embedding_json") is not None:
                embedding = record.get("embedding_json")
            embedding_json = json.dumps(embedding, ensure_ascii=False) if embedding is not None else None
            conn.execute(
                """
                INSERT OR REPLACE INTO document_chunks
                (document_id, workspace_id, chunk_index, chunk_id, text, embedding_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    workspace_id,
                    int(record.get("chunk_index", 0)),
                    chunk_id,
                    str(record.get("text", "")),
                    embedding_json,
                    metadata_json,
                    _utc_now(),
                ),
            )


def list_chunk_records(workspace_id: int) -> list[dict[str, Any]]:
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.*, d.filename, d.upload_time, d.text_content, d.metadata_json AS document_metadata_json
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.workspace_id = ?
            ORDER BY d.created_at ASC, c.chunk_index ASC, c.id ASC
            """,
            (workspace_id,),
        ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        try:
            record["metadata"] = json.loads(record.get("metadata_json") or "{}")
        except Exception:
            record["metadata"] = {}
        try:
            record["embedding"] = json.loads(record["embedding_json"]) if record.get("embedding_json") else None
        except Exception:
            record["embedding"] = None
        records.append(record)
    return records


@retry_sqlite
def save_chat_message(
    workspace_id: int,
    *,
    role: str,
    text: str,
    sources: list[dict[str, Any]] | None = None,
    provider: str | None = None,
    latency_ms: float | None = None,
    token_usage: int | None = None,
) -> None:
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (workspace_id, role, text, sources_json, provider, latency_ms, token_usage, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                role,
                text,
                json.dumps(sources or [], ensure_ascii=False),
                provider,
                latency_ms,
                token_usage,
                _utc_now(),
            ),
        )


def list_chat_messages(workspace_id: int) -> list[dict[str, Any]]:
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE workspace_id = ? ORDER BY id ASC",
            (workspace_id,),
        ).fetchall()
    messages: list[dict[str, Any]] = []
    for row in rows:
        message = dict(row)
        try:
            message["sources"] = json.loads(message.get("sources_json") or "[]")
        except Exception:
            message["sources"] = []
        messages.append(message)
    return messages


@retry_sqlite
def clear_chat_history(workspace_id: int) -> None:
    with db_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE workspace_id = ?", (workspace_id,))


@retry_sqlite
def log_event(
    user_id: int,
    *,
    workspace_id: int | None,
    event_type: str,
    provider: str | None = None,
    token_usage: int | None = None,
    latency_ms: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO analytics_events (user_id, workspace_id, event_type, provider, token_usage, latency_ms, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                workspace_id,
                event_type,
                provider,
                token_usage,
                latency_ms,
                json.dumps(details or {}, ensure_ascii=False),
                _utc_now(),
            ),
        )


def get_analytics_summary(user_id: int, workspace_id: int | None = None) -> dict[str, Any]:
    params: list[Any] = [user_id]
    workspace_clause = ""
    if workspace_id is not None:
        workspace_clause = " AND workspace_id = ?"
        params.append(workspace_id)

    with db_connection() as conn:
        uploads = conn.execute(
            f"SELECT COUNT(*) AS count FROM documents WHERE workspace_id IN (SELECT id FROM workspaces WHERE user_id = ?){workspace_clause}",
            params,
        ).fetchone()["count"]
        chats = conn.execute(
            f"SELECT COUNT(*) AS count FROM chat_messages WHERE workspace_id IN (SELECT id FROM workspaces WHERE user_id = ?){workspace_clause}",
            params,
        ).fetchone()["count"]
        ocr_events = conn.execute(
            f"SELECT COUNT(*) AS count FROM analytics_events WHERE user_id = ? AND event_type = 'ocr'{workspace_clause}",
            params,
        ).fetchone()["count"]
        provider_rows = conn.execute(
            f"SELECT provider, COUNT(*) AS count FROM analytics_events WHERE user_id = ? AND provider IS NOT NULL{workspace_clause} GROUP BY provider ORDER BY count DESC",
            params,
        ).fetchall()
        latency_rows = conn.execute(
            f"SELECT event_type, AVG(latency_ms) AS avg_latency FROM analytics_events WHERE user_id = ? AND latency_ms IS NOT NULL{workspace_clause} GROUP BY event_type",
            params,
        ).fetchall()
        token_total = conn.execute(
            f"SELECT COALESCE(SUM(token_usage), 0) AS total_tokens FROM analytics_events WHERE user_id = ?{workspace_clause}",
            params,
        ).fetchone()["total_tokens"]

    return {
        "uploads": uploads,
        "chats": chats,
        "ocr_events": ocr_events,
        "provider_usage": [{"provider": row["provider"], "count": row["count"]} for row in provider_rows],
        "avg_latency": {row["event_type"]: row["avg_latency"] for row in latency_rows},
        "token_usage": token_total,
    }


def restore_workspace_payload(workspace_id: int) -> dict[str, Any]:
    workspace = get_workspace(workspace_id)
    if not workspace:
        raise ValueError("Workspace not found.")
    documents = list_documents(workspace_id)
    chunks = list_chunk_records(workspace_id)
    chats = list_chat_messages(workspace_id)
    return {
        "workspace": workspace,
        "documents": documents,
        "chunks": chunks,
        "chats": chats,
    }
