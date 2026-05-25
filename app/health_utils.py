"""Health checks for deployment monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from config import get_settings
from database import db_health_check
from ocr_utils import check_tesseract
from rag_utils import _get_client, _get_embedder


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    component: str
    details: dict[str, Any]


def check_database_health() -> HealthStatus:
    ok, details = db_health_check()
    return HealthStatus(ok=ok, component="database", details=details)


def check_ocr_health() -> HealthStatus:
    settings = get_settings()
    ok = check_tesseract()
    return HealthStatus(
        ok=ok,
        component="ocr",
        details={"tesseract_cmd": settings.tesseract_cmd, "available": ok},
    )


def check_vector_store_health() -> HealthStatus:
    try:
        client = _get_client()
        try:
            client.delete_collection("health_check_tmp")
        except Exception:
            pass
        collection = client.create_collection(name="health_check_tmp", metadata={"hnsw:space": "cosine"})
        embedder = _get_embedder()
        embedding = embedder.encode(["health check"], show_progress_bar=False).tolist()
        collection.add(documents=["health check"], embeddings=embedding, ids=["health-check-1"])
        results = collection.query(query_embeddings=embedding, n_results=1)
        client.delete_collection("health_check_tmp")
        ok = bool(results.get("documents", [[]])[0])
        return HealthStatus(ok=ok, component="vector_store", details={"available": ok})
    except Exception as exc:
        return HealthStatus(ok=False, component="vector_store", details={"available": False, "error": str(exc)})


def get_health_report() -> dict[str, Any]:
    database = check_database_health()
    ocr = check_ocr_health()
    vector_store = check_vector_store_health()
    report = {
        "status": "ok" if database.ok and ocr.ok and vector_store.ok else "degraded",
        "components": {
            "database": asdict(database),
            "ocr": asdict(ocr),
            "vector_store": asdict(vector_store),
        },
    }
    return report
