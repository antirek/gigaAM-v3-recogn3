from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from llm_rules import refine_transcript_rules, summarize_call_rules, summarize_batch_rules
from llm_ollama import (
    refine_transcript_ollama,
    summarize_call_ollama,
)
from llm_llamacpp import (
    decide_escalation_llamacpp,
    extract_facts_llamacpp,
    extract_roles_llamacpp,
    refine_transcript_llamacpp,
    summarize_batch_llamacpp,
    summarize_call_llamacpp,
)


def _backend() -> str:
    return os.getenv("LLM_BACKEND", "rules").strip().lower()


def _fallback_to_rules() -> bool:
    v = os.getenv("LLM_FALLBACK_TO_RULES", os.getenv("OLLAMA_FALLBACK_TO_RULES", "1")).strip().lower()
    return v in {"1", "true", "yes", "on"}


def refine_transcript(transcript_text: str, *, mode: str) -> Tuple[str, List[dict], Dict[str, Any]]:
    b = _backend()
    if b in {"llamacpp", "llama.cpp", "llama"}:
        try:
            return refine_transcript_llamacpp(transcript_text, mode=mode)
        except Exception:
            if not _fallback_to_rules():
                raise
            return refine_transcript_rules(transcript_text, mode=mode)
    if b == "ollama":
        try:
            return refine_transcript_ollama(transcript_text, mode=mode)
        except Exception:
            if not _fallback_to_rules():
                raise
            return refine_transcript_rules(transcript_text, mode=mode)
    return refine_transcript_rules(transcript_text, mode=mode)


def summarize_call(call_id: str, transcript_text: str) -> Dict[str, Any]:
    b = _backend()
    if b in {"llamacpp", "llama.cpp", "llama"}:
        try:
            return summarize_call_llamacpp(call_id=call_id, transcript_text=transcript_text)
        except Exception:
            if not _fallback_to_rules():
                raise
            return summarize_call_rules(call_id=call_id, transcript_text=transcript_text)
    if b == "ollama":
        try:
            return summarize_call_ollama(call_id=call_id, transcript_text=transcript_text)
        except Exception:
            if not _fallback_to_rules():
                raise
            return summarize_call_rules(call_id=call_id, transcript_text=transcript_text)
    return summarize_call_rules(call_id=call_id, transcript_text=transcript_text)


def decide_escalation(
    *,
    transcript_text: str,
    intent: str = "",
    issues: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Supervisor escalation only (does not re-summarize the call)."""
    b = _backend()
    if b in {"llamacpp", "llama.cpp", "llama"}:
        try:
            return decide_escalation_llamacpp(
                transcript_text=transcript_text,
                intent=intent or "",
                issues=issues or [],
            )
        except Exception:
            if not _fallback_to_rules():
                raise
    return {
        "needed": False,
        "severity": "low",
        "reasons": [],
        "evidence": [],
        "summary_for_manager": "",
    }


def extract_facts(call_id: str, transcript_text: str) -> Dict[str, Any]:
    b = _backend()
    if b in {"llamacpp", "llama.cpp", "llama"}:
        try:
            return extract_facts_llamacpp(call_id=call_id, transcript_text=transcript_text)
        except Exception:
            if not _fallback_to_rules():
                raise
            return {
                "call_id": call_id,
                "phones": [],
                "addresses": [],
                "amounts": [],
                "commitments": [],
                "notes": "extract_failed_fallback",
            }
    return {
        "call_id": call_id,
        "phones": [],
        "addresses": [],
        "amounts": [],
        "commitments": [],
        "notes": f"backend_{b}_no_extract",
    }


def extract_roles(call_id: str, transcript_text: str) -> Dict[str, Any]:
    b = _backend()
    if b in {"llamacpp", "llama.cpp", "llama"}:
        try:
            return extract_roles_llamacpp(call_id=call_id, transcript_text=transcript_text)
        except Exception:
            if not _fallback_to_rules():
                raise
            return {"call_id": call_id, "speakers": [], "notes": "roles_failed_fallback"}
    return {"call_id": call_id, "speakers": [], "notes": f"backend_{b}_no_roles"}


def summarize_batch(date_hint: str, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = _backend()
    if b in {"llamacpp", "llama.cpp", "llama"}:
        try:
            return summarize_batch_llamacpp(date_hint=date_hint, summaries=summaries)
        except Exception as exc:
            if not _fallback_to_rules():
                raise
            payload = summarize_batch_rules(date_hint=date_hint, summaries=summaries)
            payload["backend"] = "rules_fallback"
            payload["fallback_error"] = f"{type(exc).__name__}: {exc}"
            return payload
    return summarize_batch_rules(date_hint=date_hint, summaries=summaries)
