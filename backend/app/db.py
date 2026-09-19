"""SQLite persistence for patient cases and their agent execution traces."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class CaseRecord(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    symptoms: Mapped[str] = mapped_column(Text)
    history: Mapped[str] = mapped_column(Text, default="")
    medications_json: Mapped[str] = mapped_column(Text, default="[]")
    assessment_json: Mapped[str] = mapped_column(Text, default="{}")
    critique_json: Mapped[str] = mapped_column(Text, default="{}")
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    trace_json: Mapped[str] = mapped_column(Text, default="[]")
    loops_run: Mapped[int] = mapped_column(default=0)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()


def save_case(session: Session, *, symptoms: str, history: str, medications: list[str],
              assessment: dict, critique: dict, evidence: list[dict], trace: list[dict],
              loops_run: int) -> CaseRecord:
    record = CaseRecord(
        symptoms=symptoms,
        history=history,
        medications_json=json.dumps(medications),
        assessment_json=json.dumps(assessment),
        critique_json=json.dumps(critique),
        evidence_json=json.dumps(evidence),
        trace_json=json.dumps(trace),
        loops_run=loops_run,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def record_to_dict(record: CaseRecord) -> dict:
    return {
        "case_id": record.id,
        "created_at": record.created_at.isoformat(),
        "symptoms": record.symptoms,
        "history": record.history,
        "medications": json.loads(record.medications_json),
        "assessment": json.loads(record.assessment_json),
        "critique": json.loads(record.critique_json),
        "evidence": json.loads(record.evidence_json),
        "trace": json.loads(record.trace_json),
        "loops_run": record.loops_run,
    }
