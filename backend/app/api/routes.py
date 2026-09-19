from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.db import get_session, record_to_dict, save_case
from app.graph import run_case
from app.ragwire.document_loader import extract_text
from app.ragwire.ingest import ingest_knowledge_base
from app.ragwire.retriever import RagWire

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    ragwire = RagWire()
    return {
        "status": "ok",
        "ragwire_ready": ragwire.is_ready(),
        "knowledge_chunks": ragwire.store.count(),
    }


@router.post("/knowledge/ingest")
def ingest_knowledge(reset: bool = False) -> dict:
    try:
        count = ingest_knowledge_base(reset=reset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ingested_chunks": count}


@router.post("/cases/analyze")
async def analyze_case(
    symptoms: str = Form(...),
    history: str = Form(""),
    medications: str = Form(""),
    files: list[UploadFile] = File(default=[]),
) -> dict:
    medications_list = [m.strip() for m in medications.split(",") if m.strip()]

    documents_text: list[str] = []
    for upload in files:
        content = await upload.read()
        try:
            text = extract_text(upload.filename or "document", content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if text.strip():
            documents_text.append(text)

    final_state = run_case(
        symptoms=symptoms,
        history=history,
        medications=medications_list,
        documents_text=documents_text,
        max_loops=settings.max_verification_loops,
    )

    assessment = final_state.get("draft_assessment", {})
    critique = final_state.get("critique", {})
    evidence = final_state.get("retrieved_evidence", [])
    trace = final_state.get("trace", [])
    loops_run = final_state.get("loop_count", 0)

    session = get_session()
    try:
        record = save_case(
            session,
            symptoms=symptoms,
            history=history,
            medications=medications_list,
            assessment=assessment,
            critique=critique,
            evidence=evidence,
            trace=trace,
            loops_run=loops_run,
        )
        case_id = record.id
    finally:
        session.close()

    return {
        "case_id": case_id,
        "assessment": assessment,
        "critique": critique,
        "evidence": evidence,
        "trace": trace,
        "loops_run": loops_run,
    }


@router.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    from app.db import CaseRecord

    session = get_session()
    try:
        record = session.get(CaseRecord, case_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Case not found")
        return record_to_dict(record)
    finally:
        session.close()
