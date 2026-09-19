"""Medication Agent: analyzes medications and potential interactions."""
from __future__ import annotations

import logging

from app.agents.medication_reference import find_interactions
from app.agents.state import GraphState, append_trace
from app.llm.client import LLMUnavailableError, get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a medication-safety assistant. Given a patient's medication list and "
    "retrieved evidence, describe considerations and interactions grounded only in "
    "the provided information. Respond with strict JSON only."
)

PROMPT_TEMPLATE = """Medications: {medications}
Known interaction rule matches: {rule_matches}
Retrieved evidence:
{evidence}

Return a JSON object with keys:
- "considerations": list of short strings, one per medication or interaction concern
"""


def medication_node(state: GraphState) -> dict:
    medications = state.get("medications", [])

    if not medications:
        update = {"medication_findings": {"considerations": [], "interactions": []}}
        update.update(append_trace(state, agent="medication", summary="No medications reported."))
        return update

    interactions = find_interactions(medications)
    evidence = state.get("retrieved_evidence", [])
    evidence_text = "\n".join(f"- ({e['source']}) {e['content'][:200]}" for e in evidence[:6]) or "none yet"

    used_llm = True
    try:
        llm = get_llm()
        findings = llm.generate_json(
            PROMPT_TEMPLATE.format(
                medications=", ".join(medications),
                rule_matches=[i["description"] for i in interactions] or "none",
                evidence=evidence_text,
            ),
            system=SYSTEM_PROMPT,
        )
        if "considerations" not in findings:
            raise ValueError("Malformed medication JSON")
    except (LLMUnavailableError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("Medication analysis falling back to heuristics: %s", exc)
        used_llm = False
        findings = {
            "considerations": [i["description"] for i in interactions]
            or [f"No known interactions found among reported medications: {', '.join(medications)}."]
        }

    findings["interactions"] = interactions

    update = {"medication_findings": findings}
    update.update(
        append_trace(
            state,
            agent="medication",
            summary=f"Checked {len(medications)} medication(s), found {len(interactions)} known interaction(s) "
            f"({'LLM' if used_llm else 'heuristic fallback'}).",
            detail=findings,
        )
    )
    return update
