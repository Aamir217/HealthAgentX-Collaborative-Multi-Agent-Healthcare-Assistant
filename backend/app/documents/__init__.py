"""Per-request document processing for uploaded patient files.

Deliberately separate from the RAGWire-backed knowledge base subsystem
(`app.ragwire`): a patient's uploaded lab report is transient, case-specific
PHI, not shared medical knowledge, so it is never ingested into the vector
store. It's parsed here, on the fly, purely for the Medical Report Agent to
read.
"""
from app.documents.loader import extract_text

__all__ = ["extract_text"]
