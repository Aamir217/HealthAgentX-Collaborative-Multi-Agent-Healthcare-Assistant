"""Pydantic models shared between the API layer and the agent graph."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceChunk(BaseModel):
    content: str
    source: str
    score: float
    query: str


class TraceEntry(BaseModel):
    step: int
    agent: str
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)
    loop_iteration: int = 0


class Critique(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    followup_query: Optional[str] = None
    notes: str = ""


class Assessment(BaseModel):
    patient_summary: str = ""
    clinical_findings: list[str] = Field(default_factory=list)
    possible_conditions: list[str] = Field(default_factory=list)
    report_insights: list[str] = Field(default_factory=list)
    medication_considerations: list[str] = Field(default_factory=list)
    risk_indicators: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)


class CaseRequest(BaseModel):
    symptoms: str
    history: str = ""
    medications: list[str] = Field(default_factory=list)


class CaseResponse(BaseModel):
    case_id: str
    assessment: Assessment
    critique: Critique
    loops_run: int
    trace: list[TraceEntry]
    evidence: list[EvidenceChunk]
