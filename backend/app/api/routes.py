from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agents.state import GraphState
from app.config import settings
from app.db import get_session, record_to_dict, save_case
from app.documents.loader import extract_text
from app.graph import run_case, stream_case
from app.ragwire.ingest import ingest_knowledge_base, ingest_uploaded_files
from app.ragwire.retriever import RagWire

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    ragwire = RagWire()
    stats = ragwire.pipeline.get_stats()
    return {
        "status": "ok",
        "ragwire_ready": ragwire.is_ready(),
        "knowledge_chunks": stats.get("total_documents", 0),
    }


@router.post("/knowledge/ingest")
def ingest_knowledge(reset: bool = False) -> dict:
    try:
        count = ingest_knowledge_base(reset=reset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ingested_chunks": count}


@router.post("/knowledge/upload")
async def upload_knowledge_documents(files: list[UploadFile] = File(...)) -> dict:
    """Adds documents uploaded from the frontend to the knowledge base.

    Works from a cold start: RAGWire auto-creates its collection on first
    use, so no prior CLI ingestion step is required before the backend or
    frontend is started.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    payload = [(upload.filename or "document", await upload.read()) for upload in files]

    try:
        stats = ingest_uploaded_files(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return stats


async def _read_uploaded_documents(files: list[UploadFile]) -> list[str]:
    documents_text: list[str] = []
    for upload in files:
        content = await upload.read()
        try:
            text = extract_text(upload.filename or "document", content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if text.strip():
            documents_text.append(text)
    return documents_text


def _persist_case(
    *, symptoms: str, history: str, medications_list: list[str], final_state: GraphState
) -> dict:
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


@router.post("/cases/analyze")
async def analyze_case(
    symptoms: str = Form(...),
    history: str = Form(""),
    medications: str = Form(""),
    files: list[UploadFile] = File(default=[]),
) -> dict:
    medications_list = [m.strip() for m in medications.split(",") if m.strip()]
    documents_text = await _read_uploaded_documents(files)

    final_state = run_case(
        symptoms=symptoms,
        history=history,
        medications=medications_list,
        documents_text=documents_text,
        max_loops=settings.max_verification_loops,
    )

    return _persist_case(
        symptoms=symptoms, history=history, medications_list=medications_list, final_state=final_state
    )


@router.post("/cases/analyze/stream")
async def analyze_case_stream(
    symptoms: str = Form(...),
    history: str = Form(""),
    medications: str = Form(""),
    files: list[UploadFile] = File(default=[]),
) -> StreamingResponse:
    """Same as /cases/analyze, but streams newline-delimited JSON events as
    each agent finishes instead of blocking until the whole workflow
    (including any Loop Engineering retries) completes.

    Each line is a JSON object with a "type" of either:
    - "progress": {"type": "progress", "trace": [<new TraceEntry dicts>]}
    - "result": {"type": "result", "data": <same shape /cases/analyze returns>}
    """
    medications_list = [m.strip() for m in medications.split(",") if m.strip()]
    documents_text = await _read_uploaded_documents(files)

    def event_stream():
        seen = 0
        final_state: GraphState = {}
        for state in stream_case(
            symptoms=symptoms,
            history=history,
            medications=medications_list,
            documents_text=documents_text,
            max_loops=settings.max_verification_loops,
        ):
            final_state = state
            trace = state.get("trace", [])
            new_entries = trace[seen:]
            seen = len(trace)
            if new_entries:
                yield json.dumps({"type": "progress", "trace": new_entries}) + "\n"

        result = _persist_case(
            symptoms=symptoms, history=history, medications_list=medications_list, final_state=final_state
        )
        yield json.dumps({"type": "result", "data": result}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


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
