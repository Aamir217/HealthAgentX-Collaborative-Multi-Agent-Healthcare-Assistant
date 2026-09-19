"""Small local reference-range table used to flag abnormal lab values.

Not exhaustive — intended to demonstrate structured, explainable extraction
from lab reports without calling any external clinical database.
"""
from __future__ import annotations

import re

# name -> (low, high, unit)
REFERENCE_RANGES: dict[str, tuple[float, float, str]] = {
    "hemoglobin": (12.0, 17.5, "g/dL"),
    "hematocrit": (36.0, 52.0, "%"),
    "wbc": (4.0, 11.0, "x10^9/L"),
    "white blood cell": (4.0, 11.0, "x10^9/L"),
    "platelets": (150.0, 450.0, "x10^9/L"),
    "glucose": (70.0, 100.0, "mg/dL"),
    "fasting glucose": (70.0, 100.0, "mg/dL"),
    "hba1c": (4.0, 5.6, "%"),
    "creatinine": (0.6, 1.3, "mg/dL"),
    "ldl": (0.0, 100.0, "mg/dL"),
    "hdl": (40.0, 100.0, "mg/dL"),
    "total cholesterol": (0.0, 200.0, "mg/dL"),
    "triglycerides": (0.0, 150.0, "mg/dL"),
    "tsh": (0.4, 4.0, "mIU/L"),
    "potassium": (3.5, 5.1, "mmol/L"),
    "sodium": (135.0, 145.0, "mmol/L"),
    "systolic bp": (90.0, 120.0, "mmHg"),
    "diastolic bp": (60.0, 80.0, "mmHg"),
    "heart rate": (60.0, 100.0, "bpm"),
    "spo2": (95.0, 100.0, "%"),
}

_VALUE_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 /\-]{2,30}?)\s*[:=]?\s*(?P<value>\d+\.?\d*)\s*"
    r"(?P<unit>mg/dL|g/dL|mmol/L|mIU/L|x10\^9/L|bpm|mmHg|%)?",
    re.IGNORECASE,
)


def extract_lab_values(text: str) -> list[dict]:
    findings = []
    for match in _VALUE_RE.finditer(text):
        name = match.group("name").strip().lower()
        if name not in REFERENCE_RANGES:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        low, high, unit = REFERENCE_RANGES[name]
        status = "normal"
        if value < low:
            status = "low"
        elif value > high:
            status = "high"
        findings.append(
            {
                "test": name,
                "value": value,
                "unit": match.group("unit") or unit,
                "reference_range": f"{low}-{high} {unit}",
                "status": status,
            }
        )
    return findings
