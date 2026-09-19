"""Thin client for a locally-hosted LLM served via Ollama.

No patient data is ever sent to an external API: all generation happens
against `OLLAMA_HOST`, which defaults to localhost. If no local LLM server
is reachable (e.g. Ollama isn't running in this environment), callers should
catch `LLMUnavailableError` and use their own rule-based fallback so the
pipeline keeps functioning end-to-end for demo/offline purposes.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import requests

from app.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the local LLM backend cannot be reached or fails to respond."""


class LocalLLMClient:
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=settings.llm_timeout_s
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailableError(
                f"Local LLM at {self.host} (model={self.model}) is unreachable: {exc}"
            ) from exc

        data = response.json()
        text = data.get("response", "").strip()
        if not text:
            raise LLMUnavailableError("Local LLM returned an empty response.")
        return text

    def generate_json(self, prompt: str, system: str = "", temperature: float = 0.1) -> dict:
        """Asks the model for strict JSON and parses it. Raises LLMUnavailableError
        if the backend is unreachable, or ValueError if the output isn't valid JSON
        (both are caught by agents, which fall back to heuristics)."""
        text = self.generate(prompt, system=system, temperature=temperature)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[-1] if cleaned.lower().startswith("json") else cleaned
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Local LLM did not return JSON: {text[:200]}")
        return json.loads(cleaned[start : end + 1])


_singleton: Optional[LocalLLMClient] = None


def get_llm() -> LocalLLMClient:
    global _singleton
    if _singleton is None:
        _singleton = LocalLLMClient()
    return _singleton
