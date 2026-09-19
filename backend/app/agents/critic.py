"""Critic/Verifier Agent: checks the draft assessment for contradictions,
unsupported claims, and hallucinations before it is allowed to finalize.

Combines a hard heuristic gate (evidence-grounding checks that always run,
regardless of LLM availability) with an optional LLM self-critique pass for
softer issues like internal contradictions.
"""
from __future__ import annotations

import logging

from app.agents.state import GraphState, append_trace
from app.llm.client import LLMUnavailableError, get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict clinical QA reviewer. Compare the draft assessment against "
    "the retrieved evidence. Flag any claim not supported by the evidence, any "
    "internal contradiction (e.g. urgency vs next steps mismatch), and any sign "
    "of hallucinated specifics (numbers, drug names, sources) not present in the "
    "evidence. Respond with strict JSON only."
)

PROMPT_TEMPLATE = """Draft assessment:
{assessment}

Retrieved evidence (source: excerpt):
{evidence}

Return a JSON object with keys:
- "contradictions": list of short strings describing internal contradictions (empty if none)
- "hallucination_risk": list of short strings describing likely unsupported/fabricated claims (empty if none)
- "notes": one-sentence overall verdict
"""


def _heuristic_grounding_check(assessment: dict, evidence: list[dict]) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    unsupported: list[str] = []

    evidence_sources = {e["source"] for e in evidence}

    substantive_claims = (
        assessment.get("possible_conditions", [])
        + assessment.get("risk_indicators", [])
        + assessment.get("report_insights", [])
    )
    if substantive_claims and not evidence:
        issues.append("Assessment makes clinical claims but no supporting evidence was retrieved.")

    for claim in assessment.get("supporting_evidence", []):
        if not any(source in claim for source in evidence_sources):
            unsupported.append(claim)

    if not assessment.get("supporting_evidence") and substantive_claims:
        issues.append("No supporting_evidence citations provided for the claims made.")

    return issues, unsupported


def critic_node(state: GraphState) -> dict:
    assessment = state.get("draft_assessment", {})
    evidence = state.get("retrieved_evidence", [])

    issues, unsupported = _heuristic_grounding_check(assessment, evidence)

    used_llm = True
    try:
        llm = get_llm()
        llm_critique = llm.generate_json(
            PROMPT_TEMPLATE.format(
                assessment=assessment,
                evidence="\n".join(f"- ({e['source']}) {e['content'][:220]}" for e in evidence[:8]) or "none",
            ),
            system=SYSTEM_PROMPT,
        )
        issues.extend(llm_critique.get("contradictions", []))
        unsupported.extend(llm_critique.get("hallucination_risk", []))
        notes = llm_critique.get("notes", "")
    except (LLMUnavailableError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("Critic LLM self-critique unavailable, using heuristic gate only: %s", exc)
        used_llm = False
        notes = "Heuristic evidence-grounding check only (LLM self-critique unavailable)."

    passed = not issues and not unsupported

    followup_query = None
    if not passed:
        candidate_conditions = state.get("symptom_findings", {}).get("candidate_conditions", [])
        covered_sources = {e["source"] for e in evidence}
        followup_query = next(
            (c for c in candidate_conditions if not any(c.lower() in e["content"].lower() for e in evidence)),
            None,
        ) or (candidate_conditions[0] if candidate_conditions else "additional supporting medical evidence")
        if not covered_sources:
            followup_query = state.get("symptoms", followup_query)

    critique = {
        "passed": passed,
        "issues": issues,
        "unsupported_claims": unsupported,
        "followup_query": followup_query,
        "notes": notes,
    }

    update = {"critique": critique}
    update.update(
        append_trace(
            state,
            agent="critic",
            summary=(
                f"Verification {'PASSED' if passed else 'FAILED'}: {len(issues)} issue(s), "
                f"{len(unsupported)} unsupported claim(s) ({'LLM+heuristic' if used_llm else 'heuristic only'})."
            ),
            detail=critique,
        )
    )
    return update
