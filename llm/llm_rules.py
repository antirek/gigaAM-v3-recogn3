from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


SPEAKER_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+Спикер\s+(?P<spk>\d+)\s*:\s*(?P<text>.*)$"
)


def _normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _cleanup_filler(s: str) -> str:
    # Light cleanup: keep discourse markers but normalize.
    s = s.replace("— ", "- ").replace("– ", "- ")
    s = re.sub(r"\s+-\s+", " - ", s)
    # normalize "э-э-э" / "э-э"
    s = re.sub(r"э-э-э+", "э-э", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r",\s+", ", ", s)
    return s


def _ensure_period(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith((".", "!", "?", "...")):
        return s
    # Don't force punctuation after trailing separators.
    if s.endswith((",", ";", ":", "-", "—")):
        return s
    # Do not force punctuation into abbreviations/phones.
    if re.search(r"\d\s*$", s):
        return s
    return s + "."


def refine_transcript_rules(transcript_text: str, *, mode: str = "safe") -> Tuple[str, List[dict], Dict[str, Any]]:
    """
    Deterministic transcript refinement.
    - safe: normalization, punctuation, whitespace, minor cleanup
    - smart: placeholder (kept for future LLM integration); currently behaves like safe
    """
    if mode not in {"safe", "smart"}:
        raise ValueError("mode must be safe|smart")

    refined_lines: List[str] = []
    edits: List[dict] = []

    for raw_line in (transcript_text or "").splitlines():
        line = raw_line.rstrip()
        m = SPEAKER_LINE_RE.match(line)
        if not m:
            # Keep non-standard lines as-is.
            if line.strip():
                refined_lines.append(_normalize_ws(_cleanup_filler(line)))
            continue

        ts = m.group("ts")
        spk = m.group("spk")
        text = m.group("text") or ""

        text2 = _normalize_ws(text)
        before = text2
        text2 = _cleanup_filler(text2)
        if text2 != before:
            edits.append({"type": "other", "count": 1})
        text2 = _ensure_period(text2)
        refined_lines.append(f"[{ts}] Спикер {spk}: {text2}")

    notes = {
        "backend": "rules",
        "mode": mode,
        "safety": "no entity rewrites; only whitespace/punctuation/cleanup",
    }
    return "\n".join(refined_lines) + ("\n" if refined_lines else ""), edits, notes


def _extract_entities(transcript_text: str) -> Dict[str, List[str]]:
    # Very lightweight entity extraction that doesn't hallucinate.
    phones = re.findall(r"\b(?:\+7|8)\s*[-(]?\d{3}[-) ]?\d{3}[- ]?\d{2,3}\b|\b\d{3}\s*[- ]?\d{3}\s*[- ]?\d{2,3}\b", transcript_text)
    emails = re.findall(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", transcript_text)
    inns = re.findall(r"\b\d{10}\b", transcript_text)
    dates = re.findall(r"\b\d{1,2}\s*(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b", transcript_text, flags=re.IGNORECASE)
    amounts = re.findall(r"\b\d[\d\s]*\s*(?:₽|руб\.|рублей)\b|\b\d+(?:[.,]\d+)?\s*(?:₽)\b", transcript_text)
    return {
        "phones": [p.strip() for p in phones][:30],
        "emails": [e.strip() for e in emails][:30],
        "inn": inns[:10],
        "dates": dates[:20],
        "amounts": amounts[:30],
    }


def _detect_intent_topics_issues(transcript_text: str) -> Tuple[str, List[str], List[dict], List[dict]]:
    t = transcript_text.lower()
    topics: List[str] = []

    def has_any(*words: str) -> bool:
        return any(w in t for w in words)

    intent = "service_request"

    # Topics
    if has_any("не оплач", "задолж", "баланс", "не оплачено", "отключ", "исключения"):
        topics.append("billing/overdue_or_balance")
        intent = "billing"
    if has_any("счет", "счёт", "сверк", "упд", "уПД", "упдк", "эдо"):
        topics.append("docs/invoices")
        intent = intent if intent else "docs"
    if has_any("контракт", "подпис", "упдк", "договора"):
        topics.append("contract")
        intent = "contract"
    if has_any("номер", "ip", "телефони", "whatsapp", "telegram", "лицензи"):
        topics.append("technical_service_or_licensing")
        intent = "technical/telephony"
    if has_any("почт", "email", "инн", "act сверки"):
        topics.append("admin/docs_delivery")

    # Issues
    issues: List[dict] = []
    if has_any("не работает", "отключ", "не подключ", "не может дозвониться", "мертв", "не дозвон", "заблокир"):
        issues.append(
            {
                "issue": "service_blocked_or_unreachable",
                "evidence": "speech about unavailability/blocked service",
                "severity": "high",
            }
        )
    if has_any("не оплач", "уведомлен", "отключ"):
        issues.append(
            {
                "issue": "billing_alert_misinterpretation",
                "evidence": "speech about overdue/unpaid invoice vs real payment",
                "severity": "med",
            }
        )

    if not issues:
        issues.append(
            {"issue": "general_service_inquiry", "evidence": "no explicit incident detected", "severity": "low"}
        )

    # Actions
    actions: List[dict] = []
    # Deadlines: "до 20 августа", "до понедельника"
    if "до 20" in t:
        actions.append({"who": "agent", "action": "verify payment timing and avoid disabling before 20th", "deadline": "20th"})
    if "до понедель" in t:
        actions.append({"who": "agent", "action": "schedule follow-up by Monday", "deadline": "Monday"})
    if "сброс" in t or "письм" in t:
        actions.append({"who": "agent", "action": "send documents by email / schedule letter", "deadline": None})
    if "техподдерж" in t:
        actions.append({"who": "support", "action": "create ticket to technical support with mentioned number", "deadline": None})

    return intent, topics, issues, actions


def summarize_call_rules(*, call_id: str, transcript_text: str) -> Dict[str, Any]:
    entities = _extract_entities(transcript_text)
    intent, topics, issues, actions = _detect_intent_topics_issues(transcript_text)

    # Transfer detection: simple cue-based
    has_transfer = bool(re.search(r"(перевед|переключ|соедин|передам)\b", transcript_text, flags=re.IGNORECASE))

    summary: Dict[str, Any] = {
        "call_id": call_id,
        "language": "ru",
        "participants": {
            "speakers": [],
            "roles_guess": {"agent": None, "client": None, "manager": None},
        },
        "intent": intent,
        "topics": topics[:8],
        "timeline": [],
        "entities": {
            "companies": [],
            "emails": entities["emails"],
            "phones": entities["phones"],
            "inn": entities["inn"],
            "dates": entities["dates"],
            "amounts": entities["amounts"],
            "addresses": [],
        },
        "actions": actions,
        "issues_detected": issues,
        "quality_notes": {
            "has_transfer": has_transfer,
            "transfer_reason": None,
            "asr_uncertainty": None,
        },
    }
    return summary


def summarize_batch_rules(*, date_hint: str, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_calls = len(summaries)
    intents: Dict[str, int] = {}
    topics: Dict[str, int] = {}
    issues: Dict[str, int] = {}

    for s in summaries:
        it = s.get("intent") or "unknown"
        intents[it] = intents.get(it, 0) + 1
        for t in s.get("topics") or []:
            topics[t] = topics.get(t, 0) + 1
        for i in s.get("issues_detected") or []:
            name = i.get("issue") or "issue"
            issues[name] = issues.get(name, 0) + 1

    def topk(d: Dict[str, int], k: int) -> List[Dict[str, Any]]:
        return [{"name": key, "count": val} for key, val in sorted(d.items(), key=lambda kv: -kv[1])[:k]]

    overall = {
        "top_intents": [{"intent": k["name"], "count": k["count"]} for k in topk(intents, 5)],
        "top_topics": [{"topic": k["name"], "count": k["count"]} for k in topk(topics, 5)],
        "top_issues": [{"issue": k["name"], "count": k["count"]} for k in topk(issues, 5)],
    }

    # Simple clustering placeholder: group by top issue
    clusters: List[dict] = []
    top_issue = overall["top_issues"][0]["issue"] if overall["top_issues"] else None
    if top_issue:
        calls = [s.get("call_id") for s in summaries if any(i.get("issue") == top_issue for i in s.get("issues_detected") or [])]
        clusters.append(
            {
                "cluster_name": str(top_issue),
                "description": "Auto cluster by most frequent issue",
                "calls": calls,
                "patterns": [],
                "recommended_fix": None,
            }
        )

    payload = {
        "date": date_hint or "",
        "n_calls": n_calls,
        "overall": overall,
        "clusters": clusters,
        "per_call": [{"call_id": s.get("call_id"), "intent": s.get("intent"), "issues": s.get("issues_detected") or []} for s in summaries],
    }
    return payload

