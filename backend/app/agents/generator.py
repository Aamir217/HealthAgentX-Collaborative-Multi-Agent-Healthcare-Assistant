"""Generator: synthesizes the structured, evidence-grounded assessment.

This is the "Generate" step of Loop Engineering (Analyze -> Retrieve ->
Generate -> Critique -> Verify -> Re-analyze/Retrieve -> Finalize). It is
re-run on every loop iteration after additional retrieval.
"""
from __future__ import annotations

import logging

from app.agents.state import GraphState, append_trace
from app.llm.client import LLMUnavailableError, get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical assessment synthesizer supporting a licensed clinician. "
    "You must ground every claim in the provided findings and evidence, and "
    "explicitly cite evidence sources by filename in 'supporting_evidence'. "
    "Never state a diagnosis with certainty; use cautious clinical language "
    "('may suggest', 'is consistent with'). If evidence is insufficient, say so "
    "rather than inventing information. Respond with strict JSON only."
)

PROMPT_TEMPLATE = """Symptoms: {symptoms}
History: {history}
Symptom analysis: {symptom_findings}
Report findings: {report_findings}
Medication findings: {medication_findings}
Risk findings: {risk_findings}
Retrieved evidence (source: excerpt):
{evidence}

Previous critique to address (if any): {critique}

Return a JSON object with exactly these keys, each a list of short strings
(except patient_summary, a single string):
- "patient_summary"
- "clinical_findings"
- "possible_conditions"
- "report_insights"
- "medication_considerations"
- "risk_indicators"
- "supporting_evidence"  (cite evidence by source filename)
- "suggested_next_steps"
"""


def _format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "none retrieved"
    return "\n".join(f"- ({e['source']}) {e['content'][:220]}" for e in evidence[:8])


def _heuristic_assessment(state: GraphState) -> dict:
    symptom_findings = state.get("symptom_findings", {})
    report_findings = state.get("report_findings", {})
    medication_findings = state.get("medication_findings", {})
    risk_findings = state.get("risk_findings", {})
    evidence = state.get("retrieved_evidence", [])

    return {
        "patient_summary": (
            f"Patient reports: {state.get('symptoms', 'n/a')}. History: {state.get('history') or 'none reported'}."
        ),
        "clinical_findings": symptom_findings.get("key_findings", []),
        "possible_conditions": symptom_findings.get("candidate_conditions", []),
        "report_insights": report_findings.get("abnormal_findings", []) or [report_findings.get("summary", "")],
        "medication_considerations": medication_findings.get("considerations", []),
        "risk_indicators": risk_findings.get("risk_indicators", []),
        "supporting_evidence": [f"({e['source']}) {e['content'][:160]}" for e in evidence[:6]],
        "suggested_next_steps": [
            "Correlate findings with a licensed clinician for definitive diagnosis.",
            "Consider further diagnostic workup for any flagged risk indicators.",
        ],
    }


def generator_node(state: GraphState) -> dict:
    evidence = state.get("retrieved_evidence", [])

    used_llm = True
    try:
        llm = get_llm()
        assessment = llm.generate_json(
            PROMPT_TEMPLATE.format(
                symptoms=state.get("symptoms", ""),
                history=state.get("history", ""),
                symptom_findings=state.get("symptom_findings", {}),
                report_findings=state.get("report_findings", {}),
                medication_findings=state.get("medication_findings", {}),
                risk_findings=state.get("risk_findings", {}),
                evidence=_format_evidence(evidence),
                critique=state.get("critique") or "none",
            ),
            system=SYSTEM_PROMPT,
        )
        required = {
            "patient_summary",
            "clinical_findings",
            "possible_conditions",
            "report_insights",
            "medication_considerations",
            "risk_indicators",
            "supporting_evidence",
            "suggested_next_steps",
        }
        if not required.issubset(assessment.keys()):
            raise ValueError("Malformed generator JSON: missing keys")
    except (LLMUnavailableError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("Generator falling back to heuristic composition: %s", exc)
        used_llm = False
        assessment = _heuristic_assessment(state)

    update = {"draft_assessment": assessment}
    update.update(
        append_trace(
            state,
            agent="generator",
            summary=f"Synthesized draft assessment grounded in {len(evidence)} evidence chunk(s) "
            f"({'LLM' if used_llm else 'heuristic fallback'}).",
            detail={"used_llm": used_llm},
        )
    )
    return update
