"""Medical RAG Agent: retrieves supporting medical knowledge via RAGWire.

Runs once per loop iteration. On the first pass it retrieves using the
queries seeded by the coordinator/symptom/report agents; on subsequent
Loop Engineering passes it additionally retrieves using the critic's
follow-up query to fill identified evidence gaps.
"""
from __future__ import annotations

import logging

from app.agents.state import GraphState, append_trace
from app.ragwire.retriever import RagWire

logger = logging.getLogger(__name__)

_ragwire: RagWire | None = None


def _get_ragwire() -> RagWire:
    global _ragwire
    if _ragwire is None:
        _ragwire = RagWire()
    return _ragwire


def medical_rag_node(state: GraphState) -> dict:
    ragwire = _get_ragwire()

    queries = list(dict.fromkeys(q for q in state.get("retrieval_queries", []) if q))
    critique = state.get("critique") or {}
    followup = critique.get("followup_query")
    if followup and followup not in queries:
        queries.append(followup)

    existing_evidence = list(state.get("retrieved_evidence", []))
    seen_keys = {(e["source"], e["content"][:80]) for e in existing_evidence}

    new_hits = 0
    queries_to_run = queries if state.get("loop_count", 0) == 0 else ([followup] if followup else [])
    for query in queries_to_run:
        for hit in ragwire.retrieve(query):
            key = (hit["source"], hit["content"][:80])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            existing_evidence.append(hit)
            new_hits += 1

    update = {"retrieved_evidence": existing_evidence, "retrieval_queries": queries}
    update.update(
        append_trace(
            state,
            agent="medical_rag",
            summary=f"Retrieved {new_hits} new evidence chunk(s) "
            f"({len(existing_evidence)} total) via RAGWire "
            f"{'(follow-up retrieval)' if state.get('loop_count', 0) > 0 else ''}".strip(),
            detail={"queries": queries_to_run, "ragwire_ready": ragwire.is_ready()},
        )
    )
    return update
