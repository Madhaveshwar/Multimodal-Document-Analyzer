"""Standalone health check endpoint for deployment monitoring."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from health_utils import get_health_report

app = FastAPI(title="Multimodal AI Health API", version="1.0.0")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(get_health_report())


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"service": "multimodal-ai-health", "status": "ok"})
