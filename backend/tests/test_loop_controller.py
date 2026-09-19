"""Regression test for Loop Engineering's retry routing.

Exercises loop_controller_node / route_after_loop_controller directly against
a minimal stand-in graph (no RAGWire/LLM involved), so it runs even fully
offline. This guards against a real bug that slipped through: the router
function's parameter was type-annotated as `state: GraphState`, and LangGraph
filters the state to that declared schema before calling a conditional-edge
function annotated with the same TypedDict used to build the graph - so a
routing signal stashed under an undeclared key (e.g. "_decision") is silently
dropped, and the graph finalizes on the first failed verification instead of
ever retrying. The fix was declaring "decision" as a real GraphState field.
"""
from langgraph.graph import END, StateGraph

from app.agents.loop_controller import (
    FINALIZE,
    RETRY,
    loop_controller_node,
    route_after_loop_controller,
)
from app.agents.state import GraphState, append_trace


def _build_test_graph(critic_fn):
    graph = StateGraph(GraphState)
    graph.add_node("medical_rag", lambda state: append_trace(state, agent="medical_rag", summary="retrieved"))
    graph.add_node("critic", critic_fn)
    graph.add_node("loop_controller", loop_controller_node)
    graph.set_entry_point("medical_rag")
    graph.add_edge("medical_rag", "critic")
    graph.add_edge("critic", "loop_controller")
    graph.add_conditional_edges(
        "loop_controller", route_after_loop_controller, {RETRY: "medical_rag", FINALIZE: END}
    )
    return graph.compile()


def test_loop_retries_until_verification_passes():
    calls = {"n": 0}

    def fake_critic(state: GraphState) -> dict:
        calls["n"] += 1
        passed = calls["n"] >= 3
        update = {
            "critique": {
                "passed": passed,
                "issues": [] if passed else ["x"],
                "unsupported_claims": [],
                "followup_query": "more evidence",
                "notes": "",
            }
        }
        update.update(append_trace(state, agent="critic", summary=f"pass={passed}"))
        return update

    compiled = _build_test_graph(fake_critic)
    final = compiled.invoke(
        {"trace": [], "step_counter": 0, "loop_count": 0, "max_loops": 3},
        config={"recursion_limit": 100},
    )

    assert calls["n"] == 3, "critic should be re-invoked on each Loop Engineering retry"
    assert final["loop_count"] == 2
    assert final["critique"]["passed"] is True
    agents_in_order = [t["agent"] for t in final["trace"]]
    assert agents_in_order.count("medical_rag") == 3
    assert agents_in_order.count("critic") == 3


def test_loop_stops_at_max_loops_when_never_passing():
    calls = {"n": 0}

    def always_fails_critic(state: GraphState) -> dict:
        calls["n"] += 1
        update = {
            "critique": {
                "passed": False,
                "issues": ["always fails"],
                "unsupported_claims": [],
                "followup_query": "more evidence",
                "notes": "",
            }
        }
        update.update(append_trace(state, agent="critic", summary="pass=False"))
        return update

    compiled = _build_test_graph(always_fails_critic)
    final = compiled.invoke(
        {"trace": [], "step_counter": 0, "loop_count": 0, "max_loops": 3},
        config={"recursion_limit": 100},
    )

    assert calls["n"] == 3, "should retry up to max_loops before giving up"
    assert final["critique"]["passed"] is False
    assert final["loop_count"] == 2
