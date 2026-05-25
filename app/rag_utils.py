"""
rag_utils.py — Lightweight RAG: embed → store → retrieve.

Uses sentence-transformers for local embeddings and ChromaDB as the
in-memory vector store. No external embedding API calls needed.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "eshwar_rag"
_EMBED_MODEL = "all-MiniLM-L6-v2"


def _get_client():
    import chromadb
    return chromadb.Client()  # ephemeral in-memory client


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_EMBED_MODEL)


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

    embedder = _get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    ids = [hashlib.md5(c.encode()).hexdigest()[:16] + str(i) for i, c in enumerate(chunks)]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    logger.info("RAG index built: %d chunks", len(chunks))
    return collection


def retrieve(collection, query: str, k: int = 5) -> List[str]:
    """
    Retrieve the top-*k* most relevant chunks for *query*.

    Returns a list of document strings (never raises on empty results).
    """
    if collection is None:
        return []
    try:
        embedder = _get_embedder()
        q_emb = embedder.encode([query], show_progress_bar=False).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=min(k, 10))
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d and d.strip()]
    except Exception as exc:
        logger.error("RAG retrieval error: %s", exc)
        return []
