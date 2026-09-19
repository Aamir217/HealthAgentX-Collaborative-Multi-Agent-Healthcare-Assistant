from app.graph import run_case


def test_run_case_end_to_end_terminates_and_produces_assessment(seeded_knowledge_base):
    final_state = run_case(
        symptoms="Chest pain and shortness of breath for 2 days, worse with exertion.",
        history="Type 2 diabetes, hypertension.",
        medications=["Lisinopril", "Potassium"],
        documents_text=["Lab Report: Potassium: 5.6 mmol/L, Glucose: 165 mg/dL"],
        max_loops=3,
    )

    assert final_state["loop_count"] <= final_state["max_loops"]

    assessment = final_state["draft_assessment"]
    for key in (
        "patient_summary",
        "clinical_findings",
        "possible_conditions",
        "report_insights",
        "medication_considerations",
        "risk_indicators",
        "supporting_evidence",
        "suggested_next_steps",
    ):
        assert key in assessment

    trace_agents = [t["agent"] for t in final_state["trace"]]
    for expected_agent in (
        "coordinator",
        "symptom_analysis",
        "medical_report",
        "medical_rag",
        "medication",
        "risk_assessment",
        "generator",
        "critic",
        "loop_controller",
    ):
        assert expected_agent in trace_agents

    # Medication agent should catch the ACE inhibitor + potassium interaction
    # from the local rule table regardless of LLM availability.
    interactions = final_state["medication_findings"]["interactions"]
    assert any("hyperkalemia" in i["description"].lower() for i in interactions)


def test_loop_terminates_without_documents_or_medications(seeded_knowledge_base):
    final_state = run_case(symptoms="Mild headache for one day.", max_loops=2)
    assert final_state["loop_count"] <= 2
    assert "draft_assessment" in final_state
