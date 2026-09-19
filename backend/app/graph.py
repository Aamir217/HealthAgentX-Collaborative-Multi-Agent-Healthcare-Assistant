"""Wires the specialized agents into a LangGraph StateGraph implementing
Loop Engineering:

    Analyze -> Retrieve -> Generate -> Critique -> Verify
             ^                                         |
             '------------ Re-analyze/Retrieve <-------'  (if verification fails)
                                                        |
                                                        v
                                                    Finalize
"""
from __future__ import annotations

from typing import Iterator

from langgraph.graph import END, StateGraph

from app.agents.coordinator import coordinator_node
from app.agents.critic import critic_node
from app.agents.generator import generator_node
from app.agents.loop_controller import FINALIZE, RETRY, loop_controller_node, route_after_loop_controller
from app.agents.medical_report import medical_report_node
from app.agents.medical_rag import medical_rag_node
from app.agents.medication import medication_node
from app.agents.risk_assessment import risk_assessment_node
from app.agents.state import GraphState
from app.agents.symptom_analysis import symptom_analysis_node


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("coordinator", coordinator_node)
    graph.add_node("symptom_analysis", symptom_analysis_node)
    graph.add_node("medical_report", medical_report_node)
    graph.add_node("medical_rag", medical_rag_node)
    graph.add_node("medication", medication_node)
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("loop_controller", loop_controller_node)

    graph.set_entry_point("coordinator")
    graph.add_edge("coordinator", "symptom_analysis")
    graph.add_edge("symptom_analysis", "medical_report")
    graph.add_edge("medical_report", "medical_rag")
    graph.add_edge("medical_rag", "medication")
    graph.add_edge("medication", "risk_assessment")
    graph.add_edge("risk_assessment", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", "loop_controller")

    graph.add_conditional_edges(
        "loop_controller",
        route_after_loop_controller,
        {RETRY: "medical_rag", FINALIZE: END},
    )

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def _build_initial_state(
    *,
    symptoms: str,
    history: str,
    medications: list[str] | None,
    documents_text: list[str] | None,
    max_loops: int,
) -> GraphState:
    return {
        "symptoms": symptoms,
        "history": history,
        "medications": medications or [],
        "documents_text": documents_text or [],
        "max_loops": max_loops,
        "loop_count": 0,
        "trace": [],
        "step_counter": 0,
    }


def run_case(
    *,
    symptoms: str,
    history: str = "",
    medications: list[str] | None = None,
    documents_text: list[str] | None = None,
    max_loops: int = 3,
) -> GraphState:
    graph = get_graph()
    initial_state = _build_initial_state(
        symptoms=symptoms,
        history=history,
        medications=medications,
        documents_text=documents_text,
        max_loops=max_loops,
    )
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})
    return final_state


def stream_case(
    *,
    symptoms: str,
    history: str = "",
    medications: list[str] | None = None,
    documents_text: list[str] | None = None,
    max_loops: int = 3,
) -> Iterator[GraphState]:
    """Same as run_case(), but yields the accumulated state after every agent
    step (including each Loop Engineering retry) instead of only the final
    result - lets a caller (e.g. a streaming API endpoint) show live
    progress through the workflow as it runs."""
    graph = get_graph()
    initial_state = _build_initial_state(
        symptoms=symptoms,
        history=history,
        medications=medications,
        documents_text=documents_text,
        max_loops=max_loops,
    )
    yield from graph.stream(
        initial_state, config={"recursion_limit": 100}, stream_mode="values"
    )
