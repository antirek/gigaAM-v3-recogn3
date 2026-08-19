"""Hybrid extract: Natasha (phones/addr/money) + GLiNER (people/orgs/cars/messengers)."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_natasha import _parse_lines, _plain_text, _spk_at, extract_facts_natasha

_DEFAULT_MODEL = os.getenv("GLINER_MODEL", "fulstock/gliner-nerel-finetuned")

_LABELS = [
    "PERSON",
    "ORGANIZATION",
    "PRODUCT",
    "PROFESSION",
    "messenger",
    "car",
]

_SURNAME_ONLY = re.compile(
    r"^[А-ЯЁ][а-яё]+(?:ов|ёв|ев|ин|ын|ский|цкой|ская|ова|ева|ина)$"
)
_NAME_CUE = re.compile(
    r"зовут|добрый\s+день|здравствуйте|менеджер|обращат|\bс\s+$",
    re.IGNORECASE,
)
_MESSENGERS = {"макс", "max", "whatsapp", "ватсап", "telegram", "телеграм", "viber", "вайбер"}
_STOP = {
    "угу",
    "да",
    "нет",
    "хорошо",
    "ага",
    "алло",
    "сейчас",
    "сегодня",
    "менеджер",
    "менеджеру",
    "менеджером",
}

_MODEL = None


def _gliner_model():
    global _MODEL
    if _MODEL is None:
        from gliner import GLiNER

        os.environ.setdefault("HF_HOME", os.getenv("GLINER_CACHE", "/tmp/hf-gliner"))
        _MODEL = GLiNER.from_pretrained(_DEFAULT_MODEL)
    return _MODEL


def _addr_tokens(addresses: Sequence[dict]) -> set:
    toks = set()
    for a in addresses:
        for w in re.findall(r"[А-Яа-яЁёA-Za-z]+", str(a.get("text") or "")):
            if len(w) >= 4:
                toks.add(w.lower())
    return toks


def _context_ok_person(plain: str, text: str) -> bool:
    if " " in text.strip():
        return True
    if _SURNAME_ONLY.match(text.strip()):
        return False
    for m in re.finditer(re.escape(text), plain):
        left = plain[max(0, m.start() - 40) : m.start()]
        right = plain[m.end() : m.end() + 40]
        if _NAME_CUE.search(left + " " + right) or _NAME_CUE.search(left):
            return True
        if re.search(r"\bс\s+$", left, re.IGNORECASE):
            return True
    return False


def _spk_for_text(segs, text: str) -> int:
    low = text.lower()
    for s in segs:
        if low in s.text.lower():
            return s.spk
    return 0


def _dedupe_keep_longest(items: List[dict], key: str = "text", *, substr: bool = True) -> List[dict]:
    items = sorted(items, key=lambda x: -len(str(x.get(key) or "")))
    kept: List[dict] = []
    for it in items:
        t = str(it.get(key) or "").strip().lower()
        if not t:
            continue
        if any(str(k.get(key) or "").lower() == t for k in kept):
            continue
        if substr and any(t != str(k.get(key) or "").lower() and t in str(k.get(key) or "").lower() for k in kept):
            continue
        kept.append(it)
    return kept


def _extract_gliner(plain: str, segs, addresses: Sequence[dict]) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    model = _gliner_model()
    raw = model.predict_entities(plain, _LABELS, threshold=0.5)
    addr_toks = _addr_tokens(addresses)

    people: List[dict] = []
    orgs: List[dict] = []
    cars: List[dict] = []
    messengers: List[dict] = []

    products: List[str] = []
    for e in raw:
        lab = str(e.get("label") or "")
        text = str(e.get("text") or "").strip()
        score = float(e.get("score") or 0)
        if lab in {"PRODUCT", "car"} and text:
            products.append(text)

    for e in raw:
        lab = str(e.get("label") or "")
        text = str(e.get("text") or "").strip()
        score = float(e.get("score") or 0)
        if not text or text.lower() in _STOP:
            continue
        item = {
            "text": text,
            "score": round(score, 3),
            "speaker": _spk_for_text(segs, text),
            "evidence": text,
        }
        low = text.lower()

        if lab == "PERSON":
            if score < 0.9:
                continue
            if " " not in text and any(low == t or t in low for t in addr_toks):
                continue
            if not _context_ok_person(plain, text):
                continue
            people.append(item)
        elif lab == "ORGANIZATION":
            if score < 0.85:
                continue
            if any(low != p.lower() and low in p.lower() for p in products):
                continue  # Toyota ⊂ Toyota Venza
            if low in {"отделом продаж", "отдел продаж"}:
                continue
            orgs.append(item)
        elif lab in {"PRODUCT", "car"}:
            if score < 0.85:
                continue
            if low in _MESSENGERS:
                messengers.append({**item, "text": text})
                continue
            if not (re.search(r"\d", text) or " " in text or lab == "car"):
                continue
            cars.append(item)
        elif lab == "messenger":
            if score >= 0.7:
                messengers.append(item)
        elif lab == "PROFESSION":
            continue

    return (
        _dedupe_keep_longest(people, substr=False)[:8],
        _dedupe_keep_longest(orgs)[:6],
        _dedupe_keep_longest(cars)[:6],
        _dedupe_keep_longest(messengers)[:4],
    )


def extract_facts_hybrid(*, call_id: str, transcript_text: str) -> Dict[str, Any]:
    base = extract_facts_natasha(call_id=call_id, transcript_text=transcript_text)
    segs = _parse_lines(transcript_text)
    plain = _plain_text(segs)
    people, orgs, cars, messengers = _extract_gliner(plain, segs, base.get("addresses") or [])
    notes = str(base.get("notes") or "")
    notes = (notes + " +gliner").strip()
    base.update(
        {
            "people": people,
            "organizations": orgs,
            "cars": cars,
            "messengers": messengers,
            "notes": notes,
        }
    )
    return base
