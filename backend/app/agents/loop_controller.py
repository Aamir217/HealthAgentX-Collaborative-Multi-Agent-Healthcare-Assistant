"""Loop Controller: implements Loop Engineering.

Decides, after each Critique/Verify pass, whether to send the case back for
additional retrieval and re-generation (Re-analyze/Retrieve) or to finalize.
The loop always terminates: either verification passes, or `max_loops` is
reached, in which case the best-effort draft is finalized with its
outstanding issues recorded transparently in the trace.
"""
from __future__ import annotations

from app.agents.state import GraphState, append_trace

RETRY = "retry"
FINALIZE = "finalize"


def loop_controller_node(state: GraphState) -> dict:
    critique = state.get("critique", {})
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)
    passed = bool(critique.get("passed"))

    if passed:
        decision = FINALIZE
        summary = f"Verification passed after {loop_count} loop iteration(s). Finalizing."
    elif loop_count + 1 >= max_loops:
        decision = FINALIZE
        summary = (
            f"Reached max verification loops ({max_loops}) without full verification. "
            "Finalizing best-effort assessment with outstanding issues recorded."
        )
    else:
        decision = RETRY
        summary = (
            f"Verification failed (loop {loop_count + 1}/{max_loops}). "
            f"Routing back for additional retrieval on: '{critique.get('followup_query')}'."
        )

    update = {
        "verification_passed": passed,
        "loop_count": loop_count + 1 if decision == RETRY else loop_count,
    }
    update.update(
        append_trace(state, agent="loop_controller", summary=summary, detail={"decision": decision})
    )
    update["_decision"] = decision
    return update


def route_after_loop_controller(state: GraphState) -> str:
    return state.get("_decision", FINALIZE)
