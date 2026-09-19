"""Symptom Analysis Agent: analyzes reported symptoms and patient history."""
from __future__ import annotations

import logging
import re

from app.agents.state import GraphState, append_trace
from app.llm.client import LLMUnavailableError, get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical symptom-analysis assistant supporting (never replacing) a "
    "licensed clinician. You never diagnose definitively; you only surface "
    "clinically plausible possibilities from symptoms and history so they can be "
    "further verified against medical evidence. Respond with strict JSON only."
)

PROMPT_TEMPLATE = """Patient-reported symptoms: {symptoms}
Patient history: {history}

Return a JSON object with keys:
- "key_findings": list of short clinical observations extracted from the symptoms/history
- "candidate_conditions": list of plausible conditions to investigate further (not a diagnosis)
- "search_terms": list of 2-4 concise medical search phrases to retrieve supporting evidence
"""

_RED_FLAG_KEYWORDS = {
    "chest pain": "possible cardiac condition",
    "shortness of breath": "possible respiratory/cardiac condition",
    "severe headache": "possible neurological condition",
    "fever": "possible infection",
    "cough": "possible respiratory infection",
    "fatigue": "possible metabolic or hematologic condition",
    "numbness": "possible neurological condition",
    "abdominal pain": "possible gastrointestinal condition",
    "dizziness": "possible cardiovascular or neurological condition",
    "palpitations": "possible cardiac arrhythmia",
}


def _heuristic_findings(symptoms: str, history: str) -> dict:
    text = f"{symptoms} {history}".lower()
    key_findings = []
    candidate_conditions = []
    search_terms = []
    for keyword, condition in _RED_FLAG_KEYWORDS.items():
        if keyword in text:
            key_findings.append(f"Reports {keyword}")
            candidate_conditions.append(condition)
            search_terms.append(keyword)
    if not key_findings:
        tokens = [t for t in re.split(r"[,.;]", symptoms) if t.strip()]
        key_findings = [t.strip() for t in tokens[:5]] or ["Non-specific symptom presentation"]
        search_terms = [symptoms[:60]] if symptoms else ["general symptom evaluation"]
    return {
        "key_findings": key_findings,
        "candidate_conditions": candidate_conditions or ["Undifferentiated presentation"],
        "search_terms": search_terms[:4] or [symptoms[:60]],
    }


def symptom_analysis_node(state: GraphState) -> dict:
    symptoms = state.get("symptoms", "")
    history = state.get("history", "")

    used_llm = True
    try:
        llm = get_llm()
        findings = llm.generate_json(
            PROMPT_TEMPLATE.format(symptoms=symptoms or "none reported", history=history or "none reported"),
            system=SYSTEM_PROMPT,
        )
        if not isinstance(findings.get("key_findings"), list):
            raise ValueError("Malformed symptom analysis JSON")
    except (LLMUnavailableError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("Symptom analysis falling back to heuristics: %s", exc)
        used_llm = False
        findings = _heuristic_findings(symptoms, history)

    queries = list(state.get("retrieval_queries", []))
    queries.extend(findings.get("search_terms", []))

    update = {
        "symptom_findings": findings,
        "retrieval_queries": queries,
    }
    update.update(
        append_trace(
            state,
            agent="symptom_analysis",
            summary=f"Identified {len(findings.get('key_findings', []))} clinical findings "
            f"({'LLM' if used_llm else 'heuristic fallback'}).",
            detail=findings,
        )
    )
    return update
