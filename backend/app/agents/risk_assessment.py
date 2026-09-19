"""Risk Assessment Agent: identifies potential risk indicators and red flags."""
from __future__ import annotations

import logging

from app.agents.state import GraphState, append_trace
from app.llm.client import LLMUnavailableError, get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical risk-triage assistant. Given symptom findings, report "
    "findings, and medication considerations, identify red flags that may need "
    "urgent attention. Be conservative: flag anything potentially serious. "
    "Respond with strict JSON only."
)

PROMPT_TEMPLATE = """Symptom findings: {symptom_findings}
Report abnormal findings: {report_findings}
Medication considerations: {medication_findings}

Return a JSON object with keys:
- "risk_indicators": list of short strings describing risk indicators / red flags
- "urgency": one of "routine", "urgent", "emergent"
"""

_HIGH_SEVERITY_INTERACTION = "high"


def _heuristic_risk(symptom_findings: dict, report_findings: dict, medication_findings: dict) -> dict:
    indicators = []
    urgency = "routine"

    abnormal = report_findings.get("abnormal_findings", [])
    if abnormal:
        indicators.extend(abnormal)
        urgency = "urgent"

    for interaction in medication_findings.get("interactions", []):
        indicators.append(f"Drug interaction risk: {interaction['description']}")
        if interaction.get("severity") == _HIGH_SEVERITY_INTERACTION:
            urgency = "emergent"

    candidate_conditions = symptom_findings.get("candidate_conditions", [])
    for condition in candidate_conditions:
        if "cardiac" in condition.lower() or "neurological" in condition.lower():
            indicators.append(f"Symptom pattern suggestive of {condition}; warrants prompt evaluation.")
            urgency = "urgent" if urgency == "routine" else urgency

    if not indicators:
        indicators = ["No acute red flags identified from available information."]

    return {"risk_indicators": indicators, "urgency": urgency}


def risk_assessment_node(state: GraphState) -> dict:
    symptom_findings = state.get("symptom_findings", {})
    report_findings = state.get("report_findings", {})
    medication_findings = state.get("medication_findings", {})

    used_llm = True
    try:
        llm = get_llm()
        findings = llm.generate_json(
            PROMPT_TEMPLATE.format(
                symptom_findings=symptom_findings,
                report_findings=report_findings.get("abnormal_findings", []),
                medication_findings=medication_findings.get("considerations", []),
            ),
            system=SYSTEM_PROMPT,
        )
        if "risk_indicators" not in findings:
            raise ValueError("Malformed risk assessment JSON")
    except (LLMUnavailableError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("Risk assessment falling back to heuristics: %s", exc)
        used_llm = False
        findings = _heuristic_risk(symptom_findings, report_findings, medication_findings)

    update = {"risk_findings": findings}
    update.update(
        append_trace(
            state,
            agent="risk_assessment",
            summary=f"Flagged {len(findings.get('risk_indicators', []))} risk indicator(s), "
            f"urgency={findings.get('urgency', 'unknown')} "
            f"({'LLM' if used_llm else 'heuristic fallback'}).",
            detail=findings,
        )
    )
    return update
