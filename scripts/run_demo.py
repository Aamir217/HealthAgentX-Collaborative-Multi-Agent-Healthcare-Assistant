#!/usr/bin/env python
"""Runs a sample patient case end-to-end through the multi-agent graph and
prints the final assessment and the agent execution / Loop Engineering trace.

Usage:
    python scripts/run_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.graph import run_case  # noqa: E402


def main() -> None:
    final_state = run_case(
        symptoms="Chest pain and shortness of breath for the past 2 days, worse with exertion.",
        history="Type 2 diabetes, hypertension. Non-smoker.",
        medications=["Lisinopril", "Metformin", "Potassium"],
        documents_text=[
            "Lab Report: Glucose: 165 mg/dL, HbA1c: 7.8%, Creatinine: 1.1 mg/dL, "
            "Potassium: 5.6 mmol/L, LDL: 145 mg/dL"
        ],
        max_loops=3,
    )

    print("\n=== FINAL ASSESSMENT ===")
    print(json.dumps(final_state.get("draft_assessment", {}), indent=2))

    print("\n=== CRITIQUE ===")
    print(json.dumps(final_state.get("critique", {}), indent=2))

    print(f"\n=== LOOPS RUN: {final_state.get('loop_count', 0)} ===")

    print("\n=== AGENT EXECUTION TRACE ===")
    for entry in final_state.get("trace", []):
        print(f"[step {entry['step']}] (loop {entry['loop_iteration']}) {entry['agent']}: {entry['summary']}")


if __name__ == "__main__":
    main()
