"""Medical Report Agent: interprets uploaded laboratory reports and documents."""
from __future__ import annotations

import logging

from app.agents.lab_reference import extract_lab_values
from app.agents.state import GraphState, append_trace
from app.llm.client import LLMUnavailableError, get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical report interpretation assistant. Given extracted lab "
    "values and raw document text, summarize noteworthy findings factually, "
    "without speculation beyond what the data supports. Respond with strict JSON only."
)

PROMPT_TEMPLATE = """Extracted lab values (test, value, unit, reference range, status):
{lab_values}

Raw document excerpt:
{document_excerpt}

Return a JSON object with keys:
- "summary": one-paragraph plain-language summary of the report
- "abnormal_findings": list of short strings describing each abnormal value and its clinical relevance
- "search_terms": list of 1-3 medical search phrases based on abnormal findings
"""


def medical_report_node(state: GraphState) -> dict:
    documents_text = state.get("documents_text", [])

    if not documents_text:
        update = {"report_findings": {"summary": "No documents provided.", "abnormal_findings": [], "search_terms": []}}
        update.update(
            append_trace(state, agent="medical_report", summary="No uploaded documents to analyze.")
        )
        return update

    combined_text = "\n\n".join(documents_text)
    lab_values = extract_lab_values(combined_text)
    abnormal = [lv for lv in lab_values if lv["status"] != "normal"]

    used_llm = True
    try:
        llm = get_llm()
        lab_summary_lines = "\n".join(
            f"- {lv['test']}: {lv['value']} {lv['unit']} (ref {lv['reference_range']}, {lv['status']})"
            for lv in lab_values
        ) or "none extracted"
        findings = llm.generate_json(
            PROMPT_TEMPLATE.format(
                lab_values=lab_summary_lines, document_excerpt=combined_text[:2000]
            ),
            system=SYSTEM_PROMPT,
        )
        if "abnormal_findings" not in findings:
            raise ValueError("Malformed medical report JSON")
    except (LLMUnavailableError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("Medical report analysis falling back to heuristics: %s", exc)
        used_llm = False
        findings = {
            "summary": f"Extracted {len(lab_values)} lab values from {len(documents_text)} document(s); "
            f"{len(abnormal)} outside reference range.",
            "abnormal_findings": [
                f"{lv['test'].title()} is {lv['status']} at {lv['value']} {lv['unit']} "
                f"(reference {lv['reference_range']})"
                for lv in abnormal
            ],
            "search_terms": [lv["test"] for lv in abnormal][:3] or [],
        }

    findings["lab_values"] = lab_values

    queries = list(state.get("retrieval_queries", []))
    queries.extend(findings.get("search_terms", []))

    update = {"report_findings": findings, "retrieval_queries": queries}
    update.update(
        append_trace(
            state,
            agent="medical_report",
            summary=f"Parsed {len(lab_values)} lab values, {len(abnormal)} abnormal "
            f"({'LLM' if used_llm else 'heuristic fallback'}).",
            detail=findings,
        )
    )
    return update
