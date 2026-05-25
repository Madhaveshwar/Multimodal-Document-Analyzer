"""
rag_utils.py — Lightweight RAG: embed → store → retrieve.

Uses sentence-transformers for local embeddings and ChromaDB as the
in-memory vector store. No external embedding API calls needed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from dataclasses import asdict, dataclass
from typing import Any, List, Sequence

from tracing_utils import traceable

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "eshwar_rag"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_MAX_CHUNKS_PER_DOCUMENT = 400
_MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    filename: str
    upload_time: str
    chunk_source: str
    page_number: int | None
    chunk_index: int
    chunk_id: str


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _score_overlap(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(len(query_tokens), 1)


@lru_cache(maxsize=1)
def _get_client():
    import chromadb
    return chromadb.Client()  # ephemeral in-memory client


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_EMBED_MODEL)


@traceable(name="Embedding Generation", run_type="tool")
def _encode_chunks(embedder, chunks: List[str]):
    return embedder.encode(chunks, show_progress_bar=False).tolist()


def _safe_metadata(record: ChunkRecord) -> dict:
    return {
        "filename": record.filename,
        "upload_time": record.upload_time,
        "chunk_source": record.chunk_source,
        "page_number": record.page_number if record.page_number is not None else -1,
        "chunk_index": record.chunk_index,
        "chunk_id": record.chunk_id,
    }


def _record_from_mapping(record: dict[str, Any]) -> ChunkRecord:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    return ChunkRecord(
        text=str(record.get("text", "")),
        filename=str(metadata.get("filename", record.get("filename", "unknown"))),
        upload_time=str(metadata.get("upload_time", record.get("upload_time", "unknown"))),
        chunk_source=str(metadata.get("chunk_source", record.get("chunk_source", "unknown"))),
        page_number=metadata.get("page_number", record.get("page_number")),
        chunk_index=int(metadata.get("chunk_index", record.get("chunk_index", 0))),
        chunk_id=str(metadata.get("chunk_id", record.get("chunk_id", ""))),
    )


def _unique_records(records: List[ChunkRecord]) -> List[ChunkRecord]:
    seen: set[str] = set()
    unique_records: List[ChunkRecord] = []
    for record in records:
        normalized = _normalize_text(record.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_records.append(record)
    return unique_records


def _chunk_document_text(
    text: str,
    *,
    filename: str,
    upload_time: str,
    chunk_source: str,
    page_number: int | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[ChunkRecord]:
    words = text.split()
    if not words:
        return []

    records: List[ChunkRecord] = []
    chunk_index = 0
    stride = max(1, chunk_size - overlap)

    for start in range(0, len(words), stride):
        chunk_words = words[start : start + chunk_size]
        chunk_text = " ".join(chunk_words).strip()
        if len(chunk_text) < _MIN_CHUNK_CHARS:
            continue
        if chunk_index >= _MAX_CHUNKS_PER_DOCUMENT:
            break

        chunk_id = hashlib.md5(
            f"{filename}|{upload_time}|{chunk_source}|{page_number}|{chunk_index}|{chunk_text}".encode()
        ).hexdigest()[:24]
        records.append(
            ChunkRecord(
                text=chunk_text,
                filename=filename,
                upload_time=upload_time,
                chunk_source=chunk_source,
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_id=chunk_id,
            )
        )
        chunk_index += 1

    return records


def _chunk_records_to_payload(records: Sequence[ChunkRecord]) -> List[dict]:
    return [asdict(record) for record in records]


def _document_text(document: dict[str, Any]) -> str:
    return str(
        document.get("text")
        or document.get("text_content")
        or document.get("content")
        or ""
    )


@traceable(name="Embedding Generation", run_type="tool")
def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    if not texts:
        return []
    embedder = _get_embedder()
    return _encode_chunks(embedder, list(texts))


def prepare_chunk_records_from_documents(documents: Sequence[dict[str, Any]]) -> List[dict[str, Any]]:
    """Prepare deduplicated chunk payloads from structured document metadata."""
    records: List[ChunkRecord] = []
    for document in documents:
        records.extend(
            _chunk_document_text(
                _document_text(document),
                filename=document.get("filename", "unknown"),
                upload_time=document.get("upload_time", "unknown"),
                chunk_source=document.get("chunk_source", document.get("filename", "unknown")),
                page_number=document.get("page_number"),
                chunk_size=int(document.get("chunk_size", 500)),
                overlap=int(document.get("overlap", 50)),
            )
        )

    records = _unique_records(records)
    payload: List[dict[str, Any]] = []
    for record in records:
        payload.append(asdict(record))
    return payload


@traceable(name="Embedding Generation", run_type="tool")
def build_index(chunks: List[str]) -> object:
    """
    Embed *chunks* and store in a fresh in-memory ChromaDB collection.

    Returns the collection object for subsequent queries.
    """
    if not chunks:
        raise ValueError("No text chunks provided to build_index.")

    client = _get_client()

    # Drop existing collection if present
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    records = _unique_records(
        [
            ChunkRecord(
                text=chunk,
                filename="unknown",
                upload_time="unknown",
                chunk_source="summary",
                page_number=None,
                chunk_index=i,
                chunk_id=hashlib.md5(f"unknown|summary|{i}|{chunk}".encode()).hexdigest()[:24],
            )
            for i, chunk in enumerate(chunks)
        ]
    )

    embedder = _get_embedder()
    embeddings = _encode_chunks(embedder, [record.text for record in records])

    ids = [record.chunk_id for record in records]
    metadatas = [_safe_metadata(record) for record in records]
    collection.add(documents=[record.text for record in records], embeddings=embeddings, ids=ids, metadatas=metadatas)
    logger.info("RAG index built: %d chunks", len(chunks))
    return collection


@traceable(name="Embedding Generation", run_type="tool")
def build_index_from_documents(documents: List[dict]) -> object:
    """Build a Chroma collection from structured document records."""
    if not documents:
        raise ValueError("No documents provided to build_index_from_documents.")

    client = _get_client()
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    records = [
        _record_from_mapping(record)
        for record in prepare_chunk_records_from_documents(documents)
    ]
    if not records:
        raise ValueError("No chunk records could be built from the provided documents.")

    embeddings = embed_texts([record.text for record in records])
    collection.add(
        documents=[record.text for record in records],
        embeddings=embeddings,
        ids=[record.chunk_id for record in records],
        metadatas=[_safe_metadata(record) for record in records],
    )
    logger.info("RAG index built from documents: %d chunks", len(records))
    return collection


@traceable(name="Embedding Generation", run_type="tool")
def build_index_from_chunk_records(records: Sequence[dict[str, Any]]) -> object:
    """Rebuild a collection from persisted chunk records and saved embeddings."""
    if not records:
        raise ValueError("No chunk records provided to build_index_from_chunk_records.")

    client = _get_client()
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    chunk_payload = [_record_from_mapping(record) for record in records]
    embeddings: List[List[float]] = []
    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[dict] = []

    for record in records:
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        embedding = record.get("embedding")
        if embedding is None:
            embedding = record.get("embedding_json")
        if embedding is None:
            continue
        if isinstance(embedding, str):
            import json
            embedding = json.loads(embedding)
        embeddings.append([float(value) for value in embedding])
        documents.append(str(record.get("text", "")))
        ids.append(str(metadata.get("chunk_id", record.get("chunk_id", hashlib.md5(str(record.get("text", "")).encode()).hexdigest()[:24]))))
        metadatas.append({
            "filename": metadata.get("filename", record.get("filename", "unknown")),
            "upload_time": metadata.get("upload_time", record.get("upload_time", "unknown")),
            "chunk_source": metadata.get("chunk_source", record.get("chunk_source", "unknown")),
            "page_number": metadata.get("page_number", record.get("page_number", -1)),
            "chunk_index": metadata.get("chunk_index", record.get("chunk_index", 0)),
            "chunk_id": metadata.get("chunk_id", record.get("chunk_id", "")),
        })

    if not documents:
        # Fall back to recalculating embeddings if saved vectors are unavailable.
        documents = [record.text for record in chunk_payload]
        embeddings = embed_texts(documents)
        ids = [record.chunk_id for record in chunk_payload]
        metadatas = [_safe_metadata(record) for record in chunk_payload]

    collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    logger.info("RAG index rebuilt from persisted chunks: %d chunks", len(documents))
    return collection


@traceable(name="RAG Retrieval", run_type="tool")
def retrieve(collection, query: str, k: int = 5) -> List[dict]:
    """
    Retrieve the top-*k* most relevant chunks for *query*.

    Returns a list of document strings (never raises on empty results).
    """
    if collection is None:
        return []
    try:
        embedder = _get_embedder()
        q_emb = embedder.encode([query], show_progress_bar=False).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=max(1, min(k, 20)), include=["documents", "metadatas", "distances"])
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        query_tokens = _tokenize(query)
        seen: set[str] = set()
        scored_results: List[dict] = []

        for index, doc in enumerate(docs):
            if not doc or not doc.strip():
                continue
            normalized = _normalize_text(doc)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)

            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            distance = float(distances[index]) if index < len(distances) and distances[index] is not None else 1.0
            semantic_score = 1.0 - max(0.0, min(distance, 1.0))
            lexical_score = _score_overlap(query_tokens, _tokenize(doc))
            combined_score = (semantic_score * 0.7) + (lexical_score * 0.3)
            scored_results.append(
                {
                    "text": doc,
                    "metadata": metadata,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "score": combined_score,
                }
            )

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[:k]
    except Exception as exc:
        logger.error("RAG retrieval error: %s", exc)
        return []


def format_sources(retrieved_chunks: List[dict]) -> List[dict]:
    """Normalize retrieved chunks for display and citation rendering."""
    sources: List[dict] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        sources.append(
            {
                "citation": index,
                "text": chunk.get("text", "") if isinstance(chunk, dict) else str(chunk),
                "filename": metadata.get("filename", "unknown"),
                "upload_time": metadata.get("upload_time", "unknown"),
                "chunk_source": metadata.get("chunk_source", "unknown"),
                "page_number": metadata.get("page_number", -1),
                "chunk_index": metadata.get("chunk_index", -1),
                "score": float(chunk.get("score", 0.0)) if isinstance(chunk, dict) else 0.0,
            }
        )
    return sources
