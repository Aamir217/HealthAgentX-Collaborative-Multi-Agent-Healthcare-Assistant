"""Small local drug-interaction rule table used by the Medication Agent."""
from __future__ import annotations

# "drugs" is a list (not a set) so these rules can be embedded directly into
# the agent trace / assessment and serialized to JSON without conversion.
KNOWN_INTERACTIONS: list[dict] = [
    {
        "drugs": ["warfarin", "aspirin"],
        "severity": "high",
        "description": "Increased bleeding risk when anticoagulants are combined with antiplatelet agents.",
    },
    {
        "drugs": ["warfarin", "ibuprofen"],
        "severity": "high",
        "description": "NSAIDs combined with warfarin significantly increase GI bleeding risk.",
    },
    {
        "drugs": ["lisinopril", "potassium"],
        "severity": "moderate",
        "description": "ACE inhibitors combined with potassium supplements can cause hyperkalemia.",
    },
    {
        "drugs": ["simvastatin", "grapefruit"],
        "severity": "moderate",
        "description": "Grapefruit can raise statin blood levels, increasing risk of myopathy.",
    },
    {
        "drugs": ["metformin", "contrast dye"],
        "severity": "high",
        "description": "Iodinated contrast can precipitate lactic acidosis in patients on metformin with renal impairment.",
    },
    {
        "drugs": ["sildenafil", "nitroglycerin"],
        "severity": "high",
        "description": "Combining PDE5 inhibitors with nitrates can cause severe, life-threatening hypotension.",
    },
    {
        "drugs": ["maoi", "ssri"],
        "severity": "high",
        "description": "Combining MAOIs with SSRIs risks serotonin syndrome.",
    },
]


def find_interactions(medications: list[str]) -> list[dict]:
    normalized = {m.strip().lower() for m in medications if m.strip()}
    hits = []
    for rule in KNOWN_INTERACTIONS:
        if set(rule["drugs"]).issubset(normalized):
            hits.append(rule)
    return hits
