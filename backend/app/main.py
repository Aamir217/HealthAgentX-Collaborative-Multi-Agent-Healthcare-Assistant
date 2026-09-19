from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db import init_db

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="HealthAgentX",
    description="Privacy-preserving multi-agent healthcare assistant "
    "(local LLM + RAGWire + LangGraph + Loop Engineering).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Idempotent; safe to run at import time so the schema exists regardless of
# how the ASGI app is launched (uvicorn, TestClient, etc.).
init_db()


@app.get("/")
def root() -> dict:
    return {"service": "HealthAgentX", "docs": "/docs"}
