"""Deterministic extraction: Natasha (addr/money) + phone digit/numeral assembler."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from natasha import AddrExtractor, MoneyExtractor, MorphVocab

from llm_rules import SPEAKER_LINE_RE


_PHONE_RE = re.compile(r"^(?:[78]\d{10}|9\d{9})$")
_TS_RE = re.compile(r"\[\d{1,2}:\d{2}\]")

_ALLOWED_ADDR_TYPES = {
    "город",
    "улица",
    "проспект",
    "переулок",
    "площадь",
    "шоссе",
    "проезд",
    "бульвар",
    "набережная",
    "дом",
    "корпус",
    "строение",
    "квартира",
    "район",
    "область",
    "край",
    "поселок",
    "посёлок",
    "деревня",
    "микрорайон",
}

_STRONG_ADDR_TYPES = {
    "город",
    "улица",
    "проспект",
    "переулок",
    "площадь",
    "шоссе",
    "проезд",
    "бульвар",
    "набережная",
    "район",
    "область",
    "край",
    "поселок",
    "посёлок",
    "деревня",
    "микрорайон",
}

_ONES = {
    "ноль": 0,
    "нуль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}
_TEENS = {
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}
_NUM_WORDS = set(_ONES) | set(_TEENS) | set(_TENS) | set(_HUNDREDS)
_FILLER = {
    "так",
    "да",
    "угу",
    "ага",
    "ну",
    "пожалуйста",
    "алло",
    "верно",
    "правильно",
    "понимаю",
}


@dataclass
class _Seg:
    ts: str
    spk: int
    text: str
    start: int
    end: int


_MORPH: Optional[MorphVocab] = None
_ADDR: Optional[AddrExtractor] = None
_MONEY: Optional[MoneyExtractor] = None


def _extractors() -> Tuple[AddrExtractor, MoneyExtractor]:
    global _MORPH, _ADDR, _MONEY
    if _MORPH is None:
        _MORPH = MorphVocab()
        _ADDR = AddrExtractor(_MORPH)
        _MONEY = MoneyExtractor(_MORPH)
    assert _ADDR is not None and _MONEY is not None
    return _ADDR, _MONEY


def _clean_phone_digits(raw: str) -> str:
    return re.sub(r"\D+", "", raw or "")


def _is_plausible_ru_phone(digits: str) -> bool:
    d = _clean_phone_digits(digits)
    if not _PHONE_RE.fullmatch(d):
        return False
    if re.search(r"0{5,}", d):
        return False
    if len(set(d[2:])) <= 2:
        return False
    return True


def _phone_grounded_in_transcript(digits: str, transcript: str) -> bool:
    plain = _TS_RE.sub(" ", transcript or "")
    compact = re.sub(r"\D+", "", plain)
    body = digits[-10:] if len(digits) >= 10 else digits
    if digits in compact or body in compact:
        return True
    covered = 0
    for tok in re.findall(r"\d{2,}", plain):
        if tok in digits or tok in body:
            covered += len(tok)
    return covered >= 6


def _parse_lines(transcript_text: str) -> List[_Seg]:
    segs: List[_Seg] = []
    pos = 0
    for raw in (transcript_text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = SPEAKER_LINE_RE.match(line)
        if m:
            text = (m.group("text") or "").strip()
            ts = m.group("ts")
            spk = int(m.group("spk"))
        else:
            text = line.strip()
            ts = ""
            spk = 0
        if segs:
            pos += 1  # newline between segments
        start = pos
        end = start + len(text)
        segs.append(_Seg(ts=ts, spk=spk, text=text, start=start, end=end))
        pos = end
    return segs


def _plain_text(segs: Sequence[_Seg]) -> str:
    return "\n".join(s.text for s in segs)


def _spk_at(segs: Sequence[_Seg], offset: int) -> int:
    for s in segs:
        if s.start <= offset < s.end or (offset == s.end and s.end == s.start):
            return s.spk
        if s.start <= offset <= s.end:
            return s.spk
    for s in segs:
        if offset < s.start:
            return s.spk
    return segs[-1].spk if segs else 0


def _evidence_at(segs: Sequence[_Seg], offset: int, fallback: str = "") -> str:
    for s in segs:
        if s.start <= offset <= s.end:
            return s.text[:160]
    return fallback[:160]


def _tokenize_num(text: str) -> List[str]:
    t = (text or "").lower().replace("ё", "е")
    t = t.replace("—", " ").replace("–", " ").replace("-", " ")
    t = re.sub(r"[^\w\d\s]+", " ", t, flags=re.UNICODE)
    return [p for p in t.split() if p]


def _consume_ru_number(tokens: Sequence[str], i: int) -> Tuple[Optional[int], int]:
    n = len(tokens)
    if i >= n:
        return None, i
    tok = tokens[i]
    if tok in _HUNDREDS:
        val = _HUNDREDS[tok]
        i += 1
        if i < n and tokens[i] in _TEENS:
            return val + _TEENS[tokens[i]], i + 1
        if i < n and tokens[i] in _TENS:
            val += _TENS[tokens[i]]
            i += 1
        if i < n and tokens[i] in _ONES:
            val += _ONES[tokens[i]]
            i += 1
        return val, i
    if tok in _TEENS:
        return _TEENS[tok], i + 1
    if tok in _TENS:
        val = _TENS[tok]
        i += 1
        if i < n and tokens[i] in _ONES:
            val += _ONES[tokens[i]]
            i += 1
        return val, i
    if tok in _ONES:
        return _ONES[tok], i + 1
    return None, i


def _consume_phone_chunk(tokens: Sequence[str], i: int) -> Tuple[Optional[str], int]:
    """Parse one dictated group, keeping leading zeros ('ноль шестьдесят четыре' → 064)."""
    n = len(tokens)
    if i >= n:
        return None, i
    zeros = 0
    j = i
    while j < n and tokens[j] in {"ноль", "нуль"}:
        zeros += 1
        j += 1
    if j < n:
        val, k = _consume_ru_number(tokens, j)
        if val is not None:
            body = str(val)
            if zeros:
                return ("0" * zeros) + body, k
            return body, k
    if zeros:
        return "0" * zeros, j
    return None, i


def _line_digit_chunks(text: str) -> Optional[List[str]]:
    tokens = _tokenize_num(text)
    if not tokens:
        return None
    chunks: List[str] = []
    used = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _FILLER:
            used += 1
            i += 1
            continue
        if tok.isdigit():
            chunks.append(tok)
            used += 1
            i += 1
            continue
        chunk, j = _consume_phone_chunk(tokens, i)
        if chunk is not None and j > i:
            chunks.append(chunk)
            used += j - i
            i = j
            continue
        i += 1
    if not chunks:
        return None
    if used < max(1, int(0.6 * len(tokens))):
        return None
    return chunks


def _assemble_chunks(chunks: Sequence[str]) -> str:
    acc = ""
    last = ""
    for d in chunks:
        if not d:
            continue
        if acc.endswith(d):
            continue
        if last and d.endswith(last) and len(d) > len(last):
            acc = acc[: -len(last)] + d
            last = d
            continue
        # Agent/client echo of a piece already in the last group: «064», then «ноль», «64».
        if last and d in last and len(d) <= len(last):
            continue
        acc += d
        last = d
    return acc


def _phones_from_runs(segs: Sequence[_Seg]) -> List[dict]:
    out: List[dict] = []
    i = 0
    while i < len(segs):
        chunks = _line_digit_chunks(segs[i].text)
        if chunks is None:
            i += 1
            continue
        run_chunks: List[str] = []
        run_segs: List[_Seg] = []
        j = i
        while j < len(segs):
            c = _line_digit_chunks(segs[j].text)
            if c is None:
                break
            run_chunks.extend(c)
            run_segs.append(segs[j])
            j += 1
        assembled = _assemble_chunks(run_chunks)
        candidates = []
        digits = _clean_phone_digits(assembled)
        if _is_plausible_ru_phone(digits):
            candidates.append(digits)
        elif len(digits) > 11:
            for k in range(0, len(digits) - 10):
                for ln in (11, 10):
                    cand = digits[k : k + ln]
                    if _is_plausible_ru_phone(cand):
                        candidates.append(cand)
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            spk = next((s.spk for s in reversed(run_segs) if _line_digit_chunks(s.text)), 0)
            evidence = " ".join(s.text for s in run_segs)[:160]
            out.append({"digits": cand, "speaker": spk, "evidence": evidence})
        i = max(j, i + 1)
    return out


def _phones_compact_regex(plain: str, segs: Sequence[_Seg]) -> List[dict]:
    out: List[dict] = []
    for m in re.finditer(
        r"(?<!\d)(?:\+7|8|7)[\s\-()]{0,3}\d{3}[\s\-()]{0,3}\d{3}[\s\-]{0,3}\d{2}[\s\-]{0,3}\d{2}(?!\d)",
        plain,
    ):
        digits = _clean_phone_digits(m.group(0))
        if not _is_plausible_ru_phone(digits):
            continue
        out.append(
            {
                "digits": digits,
                "speaker": _spk_at(segs, m.start()),
                "evidence": m.group(0)[:160],
            }
        )
    return out


def _dedupe_phones(items: List[dict], transcript_text: str) -> Tuple[List[dict], List[str]]:
    kept: List[dict] = []
    dropped: List[str] = []
    seen = set()
    for item in items:
        digits = _clean_phone_digits(str(item.get("digits") or ""))
        if digits in seen:
            continue
        if not _is_plausible_ru_phone(digits):
            if digits:
                dropped.append(digits)
            continue
        spoken = bool(item.get("_spoken"))
        if not spoken and not _phone_grounded_in_transcript(digits, transcript_text):
            dropped.append(digits)
            continue
        seen.add(digits)
        kept.append(
            {
                "digits": digits,
                "speaker": int(item.get("speaker") or 0),
                "evidence": str(item.get("evidence") or "")[:160],
            }
        )
        if len(kept) >= 4:
            break
    return kept, dropped


def _addr_label(part_type: Optional[str], value: str) -> str:
    if part_type:
        return f"{part_type} {value}".strip()
    return value.strip()


def _extract_addresses(plain: str, segs: Sequence[_Seg]) -> List[dict]:
    addr_ex, _ = _extractors()
    hits: List[Tuple[int, int, str, str]] = []  # start, stop, type, value
    for m in addr_ex(plain):
        fact = m.fact
        ptype = getattr(fact, "type", None) or None
        value = str(getattr(fact, "value", "") or "").strip()
        if not value:
            continue
        if ptype == "село":
            prev = plain[max(0, m.start - 3) : m.start].lower()
            if prev.endswith("с ") or prev.endswith("со "):
                continue
        if ptype not in _ALLOWED_ADDR_TYPES:
            continue
        prev = plain[max(0, m.start - 24) : m.start].lower()
        if ptype == "шоссе" and "направлен" in prev:
            continue
        hits.append((m.start, m.stop, ptype, value))
    if not hits:
        return []

    clusters: List[List[Tuple[int, int, str, str]]] = []
    cur = [hits[0]]
    for h in hits[1:]:
        between = plain[cur[-1][1] : h[0]]
        close = h[0] - cur[-1][1] <= 48
        same_sentence = not re.search(r"[.!?]", between)
        if close and same_sentence:
            cur.append(h)
        else:
            clusters.append(cur)
            cur = [h]
    clusters.append(cur)

    out: List[dict] = []
    seen = set()
    for cl in clusters:
        types = {h[2] for h in cl}
        if not (types & _STRONG_ADDR_TYPES) and len(cl) < 2:
            continue
        if types <= {"дом", "корпус", "строение", "квартира"}:
            continue
        text = ", ".join(_addr_label(h[2], h[3]) for h in cl)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        start = cl[0][0]
        snippet = plain[cl[0][0] : cl[-1][1]]
        out.append(
            {
                "text": text,
                "speaker": _spk_at(segs, start),
                "evidence": snippet[:160],
            }
        )
        if len(out) >= 4:
            break
    return out


def _extract_amounts(plain: str, segs: Sequence[_Seg]) -> List[dict]:
    _, money_ex = _extractors()
    out: List[dict] = []
    seen = set()
    for m in money_ex(plain):
        fact = m.fact
        amount = getattr(fact, "amount", None)
        currency = str(getattr(fact, "currency", "") or "").strip()
        if amount is None:
            continue
        if isinstance(amount, float) and amount.is_integer():
            value = str(int(amount))
        else:
            value = str(amount).replace(" ", "")
        key = (value, currency)
        if key in seen:
            continue
        seen.add(key)
        snippet = plain[m.start : m.stop]
        out.append(
            {
                "value": value,
                "currency": currency,
                "what": "",
                "speaker": _spk_at(segs, m.start),
                "evidence": snippet[:160],
            }
        )
        if len(out) >= 6:
            break
    return out


def extract_facts_natasha(*, call_id: str, transcript_text: str) -> Dict[str, Any]:
    segs = _parse_lines(transcript_text)
    plain = _plain_text(segs)
    phones_raw: List[dict] = []
    for item in _phones_from_runs(segs):
        item["_spoken"] = True
        phones_raw.append(item)
    phones_raw.extend(_phones_compact_regex(plain, segs))
    phones, dropped = _dedupe_phones(phones_raw, transcript_text)
    addresses = _extract_addresses(plain, segs)
    amounts = _extract_amounts(plain, segs)
    notes = "backend=natasha"
    if dropped:
        notes += " dropped_phones:" + ",".join(dropped)
    return {
        "call_id": call_id,
        "phones": phones,
        "addresses": addresses,
        "amounts": amounts,
        "commitments": [],
        "notes": notes,
    }
