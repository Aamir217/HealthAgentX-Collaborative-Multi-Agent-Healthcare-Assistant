"""HealthAgentX Streamlit frontend.

Talks to the FastAPI backend (default http://localhost:8000) to submit a
patient case and render the evidence-grounded assessment plus the full
multi-agent / Loop Engineering execution trace.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="HealthAgentX", page_icon="🏥", layout="wide")

st.title("🏥 HealthAgentX — Multi-Agent Healthcare Assistant")
st.caption(
    "Privacy-preserving analysis powered by a local LLM, RAGWire retrieval, "
    "LangGraph multi-agent orchestration, and Loop Engineering self-critique. "
    "For clinical decision support only — not a substitute for professional medical advice."
)

with st.sidebar:
    st.header("Backend")
    st.text_input("API URL", value=BACKEND_URL, key="backend_url", disabled=True)
    if st.button("Check health"):
        try:
            resp = requests.get(f"{BACKEND_URL}/api/health", timeout=10)
            st.json(resp.json())
        except requests.RequestException as exc:
            st.error(f"Backend unreachable: {exc}")
    if st.button("(Re)ingest knowledge base"):
        try:
            resp = requests.post(f"{BACKEND_URL}/api/knowledge/ingest", params={"reset": False}, timeout=120)
            st.success(resp.json())
        except requests.RequestException as exc:
            st.error(f"Ingestion failed: {exc}")

st.subheader("Patient Case")
col1, col2 = st.columns(2)
with col1:
    symptoms = st.text_area("Symptoms", placeholder="e.g. Chest pain and shortness of breath for 2 days...", height=120)
    history = st.text_area("Medical history", placeholder="e.g. Type 2 diabetes, hypertension...", height=100)
with col2:
    medications = st.text_area("Current medications (comma-separated)", placeholder="e.g. Lisinopril, Metformin")
    uploaded_files = st.file_uploader(
        "Lab reports / medical documents (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md"],
        accept_multiple_files=True,
    )

run = st.button("Run Multi-Agent Analysis", type="primary")

if run:
    if not symptoms.strip():
        st.warning("Please enter at least the patient's symptoms.")
    else:
        with st.spinner("Coordinating agents: analyze -> retrieve -> generate -> critique -> verify..."):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                for f in (uploaded_files or [])
            ]
            data = {"symptoms": symptoms, "history": history, "medications": medications}
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/api/cases/analyze", data=data, files=files_payload or None, timeout=300
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as exc:
                st.error(f"Analysis failed: {exc}")
                result = None

        if result:
            assessment = result["assessment"]
            critique = result["critique"]
            evidence = result["evidence"]
            trace = result["trace"]
            loops_run = result["loops_run"]

            status_color = "green" if critique.get("passed") else "orange"
            st.markdown(
                f"### Verification status: :{status_color}[{'PASSED' if critique.get('passed') else 'BEST-EFFORT (max loops reached)'}] "
                f"&nbsp;·&nbsp; Loops run: {loops_run} &nbsp;·&nbsp; Case ID: `{result['case_id']}`"
            )

            sections = [
                ("Patient Summary", assessment.get("patient_summary")),
                ("Clinical Findings", assessment.get("clinical_findings")),
                ("Possible Conditions", assessment.get("possible_conditions")),
                ("Report Insights", assessment.get("report_insights")),
                ("Medication Considerations", assessment.get("medication_considerations")),
                ("Risk Indicators", assessment.get("risk_indicators")),
                ("Supporting Evidence", assessment.get("supporting_evidence")),
                ("Suggested Next Steps", assessment.get("suggested_next_steps")),
            ]
            for title, content in sections:
                st.markdown(f"#### {title}")
                if isinstance(content, list):
                    if content:
                        for item in content:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown("_none_")
                else:
                    st.write(content or "_none_")

            with st.expander("🔎 Retrieved Evidence (RAGWire)"):
                if evidence:
                    for e in evidence:
                        st.markdown(f"**{e['source']}** (score: {e['score']}, query: _{e['query']}_)")
                        st.caption(e["content"])
                else:
                    st.write("No evidence retrieved.")

            with st.expander("🧭 Agent Execution & Loop Engineering Trace", expanded=True):
                for entry in trace:
                    st.markdown(
                        f"**Step {entry['step']} · loop {entry['loop_iteration']} · `{entry['agent']}`** — {entry['summary']}"
                    )

            if critique.get("issues") or critique.get("unsupported_claims"):
                with st.expander("⚠️ Outstanding Critique Issues"):
                    for issue in critique.get("issues", []):
                        st.markdown(f"- {issue}")
                    for claim in critique.get("unsupported_claims", []):
                        st.markdown(f"- Unsupported claim: {claim}")
