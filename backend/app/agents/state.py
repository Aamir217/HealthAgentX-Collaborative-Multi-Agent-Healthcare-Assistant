"""Shared LangGraph state for the HealthAgentX multi-agent pipeline."""
from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # --- inputs ---
    symptoms: str
    history: str
    medications: list[str]
    documents_text: list[str]  # extracted text per uploaded document

    # --- agent outputs ---
    plan: dict[str, Any]
    symptom_findings: dict[str, Any]
    report_findings: dict[str, Any]
    retrieval_queries: list[str]
    retrieved_evidence: list[dict[str, Any]]
    medication_findings: dict[str, Any]
    risk_findings: dict[str, Any]
    draft_assessment: dict[str, Any]
    critique: dict[str, Any]

    # --- loop engineering / control ---
    loop_count: int
    max_loops: int
    verification_passed: bool
    # Routing signal read by route_after_loop_controller(). Must be declared
    # here (not just stashed into the update dict ad hoc): a conditional-edge
    # function whose parameter is annotated with this same GraphState type
    # only sees keys declared in the schema - LangGraph filters the state to
    # match it before calling the router, silently dropping anything else.
    decision: str

    # --- observability ---
    trace: list[dict[str, Any]]
    step_counter: int


def append_trace(state: GraphState, agent: str, summary: str, detail: dict | None = None) -> dict:
    trace = list(state.get("trace", []))
    step = state.get("step_counter", 0) + 1
    trace.append(
        {
            "step": step,
            "agent": agent,
            "summary": summary,
            "detail": detail or {},
            "loop_iteration": state.get("loop_count", 0),
        }
    )
    return {"trace": trace, "step_counter": step}
