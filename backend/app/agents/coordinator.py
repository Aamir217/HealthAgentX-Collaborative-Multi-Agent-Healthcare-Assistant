"""Coordinator Agent: plans the workflow and seeds the initial retrieval query."""
from __future__ import annotations

from app.agents.state import GraphState, append_trace


def coordinator_node(state: GraphState) -> dict:
    symptoms = state.get("symptoms", "").strip()
    history = state.get("history", "").strip()
    medications = state.get("medications", [])
    has_documents = bool(state.get("documents_text"))

    plan_steps = [
        "symptom_analysis",
        "medical_report" if has_documents else "medical_report (skipped: no documents)",
        "medical_rag",
        "medication",
        "risk_assessment",
        "generator",
        "critic",
        "loop_controller",
    ]
    plan = {
        "steps": plan_steps,
        "has_documents": has_documents,
        "medication_count": len(medications),
    }

    seed_query = symptoms or history or "general clinical assessment"

    update = {
        "plan": plan,
        "loop_count": 0,
        "max_loops": state.get("max_loops", 3),
        "retrieval_queries": [seed_query],
        "retrieved_evidence": [],
        "verification_passed": False,
    }
    update.update(
        append_trace(
            state,
            agent="coordinator",
            summary=f"Planned workflow across {len(plan_steps)} agents.",
            detail=plan,
        )
    )
    return update
