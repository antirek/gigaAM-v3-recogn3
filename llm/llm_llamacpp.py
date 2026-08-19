from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from openai import OpenAI
from pydantic import BaseModel

from llm_schemas import (
    CallSummaryResponse,
    ExtractFactsResponse,
    ExtractRolesResponse,
    SafeRefineResponse,
    SmartRefineResponse,
)


def _llamacpp_url() -> str:
    return os.getenv("LLAMACPP_URL", "http://llamacpp:8000/v1").rstrip("/")


def _llamacpp_model() -> str:
    return os.getenv("LLAMACPP_MODEL", "T-lite-it-2.1-q8_0").strip()


def _extract_json(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start : end + 1])
    raise ValueError("failed to parse llama.cpp JSON response")


_TRANSCRIPT_LINE_RE = re.compile(r"\[\d{2}:\d{2}\]\s+Спикер\s+\d+\s*:")
_SPEAKER_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+Спикер\s+(?P<spk>\d+)\s*:\s*(?P<text>.*)$"
)


def _normalize_refined_transcript(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    # If model collapsed all turns into one line, split by timestamp+speaker markers.
    # Keep existing line structure when already multiline.
    if "\n" not in s and len(_TRANSCRIPT_LINE_RE.findall(s)) > 1:
        parts = re.split(r"(?=\[\d{2}:\d{2}\]\s+Спикер\s+\d+\s*:)", s)
        lines = [p.strip() for p in parts if p.strip()]
        return "\n".join(lines)

    # Normalize accidental extra spaces around line breaks.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines)


def _parse_transcript_rows(transcript_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in (transcript_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if not m:
            rows.append({"raw": True, "text": line})
            continue
        rows.append(
            {
                "raw": False,
                "ts": m.group("ts"),
                "spk": int(m.group("spk")),
                "text": m.group("text") or "",
            }
        )
    return rows


def _format_transcript_rows(rows: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for row in rows:
        if row.get("raw"):
            out.append(str(row.get("text") or "").strip())
            continue
        text = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
        out.append(f"[{row['ts']}] Спикер {row['spk']}: {text}")
    return "\n".join(out) + ("\n" if out else "")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u00a0", " ")).strip()


def _cut_once(src: str, needle: str) -> Tuple[str, bool]:
    """Remove the first occurrence of needle, allowing whitespace differences."""
    src = src or ""
    needle = _norm_ws(needle)
    if not needle:
        return src, False
    def _tidy(s: str) -> str:
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        s = re.sub(r"([.!?])(?:\s*[.!?])+", r"\1", s)
        s = re.sub(r",\s*[.!?]+$", ".", s)
        return s.strip(" ,;:-")

    if needle in src:
        return _tidy(src.replace(needle, " ", 1)), True
    parts = [re.escape(p) for p in needle.split()]
    if not parts:
        return src, False
    m = re.search(r"\s+".join(parts), src)
    if not m:
        return src, False
    return _tidy(src[: m.start()] + " " + src[m.end() :]), True


def _next_row_index(
    rows: List[Dict[str, Any]],
    src_i: int,
    *,
    spk: int,
    ts: str = "",
) -> int | None:
    if ts:
        for j in range(src_i + 1, len(rows)):
            row = rows[j]
            if row.get("raw"):
                continue
            if row["ts"] == ts and row["spk"] == spk:
                return j
    for j in range(src_i + 1, len(rows)):
        row = rows[j]
        if not row.get("raw") and row["spk"] == spk:
            return j
    return None


def _apply_overlap_moves(
    rows: List[Dict[str, Any]],
    moves: List[dict],
) -> Tuple[List[Dict[str, Any]], List[dict], List[dict]]:
    applied: List[dict] = []
    skipped: List[dict] = []
    ordered = sorted(
        [mv for mv in moves if isinstance(mv, dict)],
        key=lambda mv: -len(_norm_ws(str(mv.get("text") or ""))),
    )
    for mv in ordered:
        text = _norm_ws(str(mv.get("text") or ""))
        if not text:
            skipped.append({**mv, "reason": "empty_text"})
            continue
        from_ts = str(mv.get("from_ts") or "").strip()
        to_ts = str(mv.get("to_ts") or "").strip()
        try:
            from_spk = int(mv.get("from_spk"))
            to_spk = int(mv.get("to_spk"))
        except (TypeError, ValueError):
            skipped.append({**mv, "reason": "bad_speakers"})
            continue

        src_i = None
        for i, row in enumerate(rows):
            if row.get("raw"):
                continue
            if row["ts"] == from_ts and row["spk"] == from_spk:
                _, found = _cut_once(str(row.get("text") or ""), text)
                if found:
                    src_i = i
                    break
        if src_i is None:
            skipped.append({**mv, "reason": "source_not_found"})
            continue

        dst_i = _next_row_index(rows, src_i, spk=to_spk, ts=to_ts)
        if dst_i is None:
            skipped.append({**mv, "reason": "dest_not_found"})
            continue

        new_src, ok = _cut_once(str(rows[src_i]["text"] or ""), text)
        if not ok:
            skipped.append({**mv, "reason": "cut_failed"})
            continue
        rows[src_i]["text"] = new_src
        dst_text = _norm_ws(str(rows[dst_i]["text"] or ""))
        kind = str(mv.get("kind") or ("stem" if from_spk == to_spk else "foreign"))
        if text not in dst_text:
            if kind == "stem" and dst_text:
                cap = text[:1].upper() + text[1:] if text else text
                m = re.match(r"^([А-ЯЁA-Z][а-яёa-z]+)(\b.*)$", dst_text)
                rest = (
                    m.group(1)[0].lower() + m.group(1)[1:] + m.group(2)
                    if m
                    else dst_text
                )
                rows[dst_i]["text"] = f"{cap} {rest}".strip()
            else:
                rows[dst_i]["text"] = f"{text} {dst_text}".strip() if dst_text else text
        applied.append(
            {
                "type": "overlap",
                "from_ts": from_ts,
                "from_spk": from_spk,
                "to_ts": rows[dst_i]["ts"],
                "to_spk": to_spk,
                "text": text,
                "source": str(mv.get("source") or "llm"),
            }
        )
    return rows, applied, skipped


_TRAILING_STEM_RE = re.compile(r"^[А-ЯЁA-Z][а-яёa-z]{0,5}$")
_TAG_Q = {"да", "нет", "так", "верно", "ага", "угу", "ладно"}
_QSTART_RE = re.compile(
    r"^(где|как|кто|что|чем|когда|какой|какая|какое|какие|почему|зачем|куда|откуда|сколько)\b",
    re.IGNORECASE,
)


def _first_content_question(text: str) -> int:
    for m in re.finditer(r"\?", text):
        before = text[: m.start()].rstrip()
        last = re.sub(r"[^\wа-яёА-ЯЁ]+$", "", before.split()[-1] if before.split() else "").lower()
        if last in _TAG_Q:
            continue
        return m.start()
    return -1


def _heuristic_overlap_moves(rows: List[Dict[str, Any]]) -> List[dict]:
    """If a question is followed by extra words in the same line, those words are likely the other speaker."""
    moves: List[dict] = []
    for i, row in enumerate(rows):
        if row.get("raw"):
            continue
        text = _norm_ws(str(row.get("text") or ""))
        q = _first_content_question(text)
        if q < 0:
            continue
        rest = text[q + 1 :].strip(" ,;:-")
        if not rest:
            continue
        words = rest.split()
        stem = ""
        foreign = rest
        if len(words) >= 2 and _TRAILING_STEM_RE.fullmatch(words[-1]):
            stem = words[-1]
            foreign = " ".join(words[:-1]).strip(" ,;:-")
        if len(foreign.split()) < 4:
            continue
        if _QSTART_RE.match(foreign):
            continue
        other_i = None
        self_i = None
        for j in range(i + 1, len(rows)):
            nxt = rows[j]
            if nxt.get("raw"):
                continue
            if other_i is None and nxt["spk"] != row["spk"]:
                other_i = j
            if self_i is None and nxt["spk"] == row["spk"]:
                self_i = j
            if other_i is not None and self_i is not None:
                break
        if other_i is None:
            continue
        other = rows[other_i]
        moves.append(
            {
                "from_ts": row["ts"],
                "from_spk": row["spk"],
                "to_ts": other["ts"],
                "to_spk": other["spk"],
                "text": foreign,
                "source": "heuristic",
            }
        )
        if stem and self_i is not None:
            self_row = rows[self_i]
            moves.append(
                {
                    "from_ts": row["ts"],
                    "from_spk": row["spk"],
                    "to_ts": self_row["ts"],
                    "to_spk": self_row["spk"],
                    "text": stem,
                    "kind": "stem",
                    "source": "heuristic",
                }
            )
    return moves


_ECHO_SKIP = {"хорошо", "понятно", "ладно", "спасибо", "алло"}
_STEM_WORDS = {"сейчас", "как"}


def _bare_phrase(s: str) -> str:
    s = _norm_ws(s)
    s = re.sub(r"[.!?…]+$", "", s).strip()
    return s


def _echo_overlap_moves(rows: List[Dict[str, Any]]) -> List[dict]:
    """If the next other-speaker line is a short phrase already glued into the current line, cut it out."""
    moves: List[dict] = []
    for i, row in enumerate(rows):
        if row.get("raw"):
            continue
        other_i = None
        for j in range(i + 1, min(i + 4, len(rows))):
            nxt = rows[j]
            if nxt.get("raw"):
                continue
            if nxt["spk"] != row["spk"]:
                other_i = j
                break
        if other_i is None:
            continue
        other = rows[other_i]
        phrase = _bare_phrase(str(other.get("text") or ""))
        words = phrase.split()
        if not (1 <= len(words) <= 3) or len(phrase) < 8:
            continue
        if phrase.lower() in _ECHO_SKIP:
            continue
        src = str(row.get("text") or "")
        _, found = _cut_once(src, phrase)
        if not found:
            continue
        src_bare = _bare_phrase(src)
        if src_bare.lower() == phrase.lower():
            continue
        moves.append(
            {
                "from_ts": row["ts"],
                "from_spk": row["spk"],
                "to_ts": other["ts"],
                "to_spk": other["spk"],
                "text": phrase,
                "kind": "foreign",
                "source": "heuristic-echo",
            }
        )
    return moves


def _stem_attach_moves(rows: List[Dict[str, Any]]) -> List[dict]:
    """Move a hanging leftover word onto the next same-speaker line."""
    moves: List[dict] = []
    for i, row in enumerate(rows):
        if row.get("raw"):
            continue
        parts = _norm_ws(str(row.get("text") or "")).split()
        if not parts:
            continue
        last = re.sub(r"[^\wа-яёА-ЯЁ]+$", "", parts[-1])
        if last.lower() not in _STEM_WORDS:
            continue
        self_i = _next_row_index(rows, i, spk=int(row["spk"]))
        if self_i is None:
            continue
        nxt = _norm_ws(str(rows[self_i].get("text") or ""))
        if nxt.lower().startswith(last.lower()):
            continue
        moves.append(
            {
                "from_ts": row["ts"],
                "from_spk": row["spk"],
                "to_ts": rows[self_i]["ts"],
                "to_spk": row["spk"],
                "text": last,
                "kind": "stem",
                "source": "heuristic-stem",
            }
        )
    return moves


def _collect_heuristic_moves(rows: List[Dict[str, Any]]) -> List[dict]:
    return _heuristic_overlap_moves(rows) + _echo_overlap_moves(rows) + _stem_attach_moves(rows)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _apply_replacements(text: str, replacements: List[dict]) -> Tuple[str, List[dict]]:
    known = {w.lower() for w in re.findall(r"[А-ЯЁа-яёA-Za-z]{4,}", text)}
    edits: List[dict] = []
    out = text
    for item in replacements:
        if not isinstance(item, dict):
            continue
        src = str(
            item.get("from")
            or item.get("src")
            or item.get("from_text")
            or ""
        ).strip()
        dst = str(
            item.get("to")
            or item.get("dst")
            or item.get("to_text")
            or ""
        ).strip()
        if not src or not dst or src == dst or src not in out:
            continue
        if any(ch.isdigit() for ch in src) or any(ch.isdigit() for ch in dst):
            continue
        allowed = dst.lower() in known or _edit_distance(src.lower(), dst.lower()) <= 2
        if not allowed:
            continue
        out = out.replace(src, dst)
        edits.append({"type": "spelling", "from": src, "to": dst})
    return out, edits


def _splits_to_moves(splits: List[Any]) -> List[dict]:
    moves: List[dict] = []
    for item in splits:
        if not isinstance(item, dict):
            continue
        from_ts = str(item.get("from_ts") or "").strip()
        try:
            from_spk = int(item.get("from_spk"))
        except (TypeError, ValueError):
            continue
        foreign = _norm_ws(str(item.get("foreign") or item.get("text") or ""))
        stem = _norm_ws(str(item.get("stem") or ""))
        if foreign:
            try:
                to_spk = int(item.get("foreign_to_spk") or item.get("to_spk"))
            except (TypeError, ValueError):
                continue
            moves.append(
                {
                    "from_ts": from_ts,
                    "from_spk": from_spk,
                    "to_ts": str(item.get("foreign_to_ts") or item.get("to_ts") or "").strip(),
                    "to_spk": to_spk,
                    "text": foreign,
                    "source": "llm",
                }
            )
        if stem:
            try:
                stem_spk = int(item.get("stem_to_spk") or from_spk)
            except (TypeError, ValueError):
                stem_spk = from_spk
            moves.append(
                {
                    "from_ts": from_ts,
                    "from_spk": from_spk,
                    "to_ts": str(item.get("stem_to_ts") or "").strip(),
                    "to_spk": stem_spk,
                    "text": stem,
                    "kind": "stem",
                    "source": "llm",
                }
            )
    return moves


def _merge_moves(llm_moves: List[dict], heuristic_moves: List[dict]) -> List[dict]:
    """Union of LLM and heuristic moves. Longer exact cuts win at apply time."""
    out: List[dict] = []
    seen = set()
    for mv in list(llm_moves) + list(heuristic_moves):
        try:
            key = (
                str(mv.get("from_ts") or ""),
                int(mv["from_spk"]),
                int(mv["to_spk"]),
                _norm_ws(str(mv.get("text") or "")),
            )
        except (TypeError, ValueError, KeyError):
            continue
        if not key[3] or key in seen:
            continue
        seen.add(key)
        out.append(mv)
    return out


_OPENAI_CLIENTS: Dict[str, OpenAI] = {}


def _openai() -> OpenAI:
    url = _llamacpp_url()
    client = _OPENAI_CLIENTS.get(url)
    if client is None:
        client = OpenAI(
            base_url=url,
            api_key=os.getenv("LLAMACPP_API_KEY", "not-needed"),
            timeout=float(os.getenv("LLAMACPP_REQUEST_TIMEOUT_SEC", "180")),
        )
        _OPENAI_CLIENTS[url] = client
    return client


def _is_truncation_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        "truncated" in msg
        or "lengthfinish" in name.lower()
        or "length limit" in msg
        or "eof while parsing" in msg
        or "json_invalid" in msg
        or "unterminated" in msg
    )


def _chat_json_once(
    *,
    system: str,
    user: str,
    max_tokens: int,
    schema: Optional[Type[BaseModel]],
    schema_name: str,
) -> Dict[str, Any]:
    client = _openai()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    errors: List[str] = []
    model = _llamacpp_model()

    if schema is not None:
        json_schema = schema.model_json_schema()
        formats = [
            (
                "json_schema",
                {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
                },
                None,
            ),
            (
                "json_schema_flat",
                {"type": "json_schema", "schema": json_schema},
                None,
            ),
            (
                "json_object_schema",
                {"type": "json_object", "schema": json_schema},
                None,
            ),
            (
                "extra_json_schema",
                {"type": "json_object"},
                {"json_schema": json_schema},
            ),
        ]
        for label, response_format, extra in formats:
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "response_format": response_format,
                }
                if extra:
                    kwargs["extra_body"] = extra
                completion = client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content
                finish = getattr(completion.choices[0], "finish_reason", None)
                if finish == "length":
                    raise ValueError(f"length limit was reached ({max_tokens} tokens)")
                data = schema.model_validate_json(content or "").model_dump()
                data["_llm_format"] = label
                return data
            except Exception as exc:
                errors.append(f"{label} {type(exc).__name__}: {exc}")
                if _is_truncation_error(exc):
                    raise RuntimeError("llama.cpp structured JSON truncated: " + " || ".join(errors)) from exc
        raise RuntimeError("llama.cpp structured JSON failed: " + " || ".join(errors))

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    finish = getattr(completion.choices[0], "finish_reason", None)
    if finish == "length":
        raise ValueError(f"length limit was reached ({max_tokens} tokens)")
    data = _extract_json(completion.choices[0].message.content)
    data.setdefault("_llm_format", "json_object")
    return data


def _chat_json(
    *,
    system: str,
    user: str,
    max_tokens: int = 1024,
    schema: Optional[Type[BaseModel]] = None,
    schema_name: str = "response",
) -> Dict[str, Any]:
    cap = int(os.getenv("LLAMACPP_MAX_TOKENS", "2048"))
    first = min(int(max_tokens), cap)
    attempts = [first]
    if cap > first:
        attempts.append(cap)
    last_exc: Optional[BaseException] = None
    for n in attempts:
        try:
            return _chat_json_once(
                system=system,
                user=user,
                max_tokens=n,
                schema=schema,
                schema_name=schema_name,
            )
        except Exception as exc:
            last_exc = exc
            if n == attempts[-1] or not _is_truncation_error(exc):
                raise
    raise RuntimeError(str(last_exc))


def refine_transcript_llamacpp(transcript_text: str, *, mode: str = "safe") -> Tuple[str, List[dict], Dict[str, Any]]:
    if mode == "smart":
        return _refine_transcript_llamacpp_smart(transcript_text)
    return _refine_transcript_llamacpp_safe(transcript_text)


def _refine_transcript_llamacpp_safe(transcript_text: str) -> Tuple[str, List[dict], Dict[str, Any]]:
    system = (
        "Ты редактор русской телефонной расшифровки. Верни только JSON по схеме. "
        "Не выдумывай имена и числа. Не переноси текст между спикерами."
    )
    user = (
        "SAFE: только явные опечатки ASR (1–2 буквы) или слово, которое уже есть в этом тексте.\n"
        "Пустой replacements допустим. Новые таймкоды не создавать.\n\n"
        f"{transcript_text}\n"
    )
    payload = _chat_json(
        system=system,
        user=user,
        max_tokens=1024,
        schema=SafeRefineResponse,
        schema_name="safe_refine",
    )
    llm_format = payload.pop("_llm_format", "")
    repls = payload.get("replacements") or []
    if not isinstance(repls, list):
        repls = []
    refined, spell_edits = _apply_replacements(transcript_text, repls)
    from llm_rules import refine_transcript_rules

    refined, punct_edits, _notes = refine_transcript_rules(refined, mode="safe")
    safety = payload.get("safety") or {}
    if isinstance(safety, str):
        safety = {"mode": "safe", "notes": safety}
    elif not isinstance(safety, dict):
        safety = {"mode": "safe", "notes": str(safety)}
    safety["backend"] = "llamacpp"
    safety["mode"] = "safe"
    if llm_format:
        safety["llm_format"] = llm_format
    edits = spell_edits + [e for e in punct_edits if isinstance(e, dict)]
    return refined, edits, safety


def _refine_transcript_llamacpp_smart(transcript_text: str) -> Tuple[str, List[dict], Dict[str, Any]]:
    rows = _parse_transcript_rows(transcript_text)
    heuristic = _collect_heuristic_moves(rows)
    numbered = "\n".join(
        f"{i+1}. [{r['ts']}] Спикер {r['spk']}: {r['text']}" if not r.get("raw") else f"{i+1}. {r['text']}"
        for i, r in enumerate(rows)
    )
    system = (
        "Ты редактор русской телефонной расшифровки. Верни только JSON. "
        "Не выдумывай текст: keep/foreign/stem — точные куски исходной строки. "
        "Если склеек нет — splits: []. Не больше 6 сплитов, не описывай каждую строку."
    )
    user = (
        "SMART: ASR часто склеивает речь двух спикеров в одну строку.\n"
        "1) Чужой ответ после вопроса в той же строке — вырежи и приклей к СЛЕДУЮЩЕЙ строке другого спикера.\n"
        "2) Эхо: короткая следующая реплика B целиком торчит внутри текущей строки A "
        "(пример: «Здравствуйте» у клиента, а следом та же реплика у менеджера) — вырежи из A.\n"
        "3) Обрывок A в конце строки («Как», «сейчас») — клей к следующей строке того же спикера.\n"
        "4) replacements: только явные опечатки ASR. src — точная подстрока, "
        "dst — исправление. Не выдумывай имена и адреса: править можно, если слово "
        "уже встречается в этом же транскрипте или это 1–2 буквы опечатки.\n"
        "Новые таймкоды не создавать.\n\n"
        "Кандидаты от эвристики:\n"
        f"{json.dumps(heuristic, ensure_ascii=False)}\n\n"
        "Нумерованный транскрипт:\n"
        f"{numbered}\n"
    )
    try:
        payload = _chat_json(
            system=system,
            user=user,
            max_tokens=1024,
            schema=SmartRefineResponse,
            schema_name="smart_refine",
        )
        llm_format = payload.pop("_llm_format", "json_schema")
    except Exception as exc:
        llm_format = ""
        payload = {
            "splits": [],
            "replacements": [],
            "safety": {"notes": f"llm_json_failed: {exc}"},
        }
    splits = payload.get("splits") or []
    if not isinstance(splits, list):
        splits = []
    valid_splits: List[dict] = []
    invalid_splits: List[dict] = []
    for item in splits:
        if not isinstance(item, dict):
            continue
        src = ""
        try:
            from_spk = int(item.get("from_spk"))
        except (TypeError, ValueError):
            invalid_splits.append(item)
            continue
        from_ts = str(item.get("from_ts") or "").strip()
        for row in rows:
            if not row.get("raw") and row["ts"] == from_ts and row["spk"] == from_spk:
                src = str(row.get("text") or "")
                break
        keep = _norm_ws(str(item.get("keep") or ""))
        foreign = _norm_ws(str(item.get("foreign") or item.get("text") or ""))
        stem = _norm_ws(str(item.get("stem") or ""))
        recon = _norm_ws(" ".join(x for x in (keep, foreign, stem) if x))
        if src and recon == _norm_ws(src) and foreign:
            valid_splits.append(item)
        elif src and foreign:
            _, found = _cut_once(src, foreign)
            if found:
                valid_splits.append(item)
            else:
                invalid_splits.append(item)
        elif src and stem:
            _, found = _cut_once(src, stem)
            if found:
                valid_splits.append(item)
            else:
                invalid_splits.append(item)
        else:
            invalid_splits.append(item)
    llm_moves = _splits_to_moves(valid_splits)
    if not llm_moves:
        raw_moves = payload.get("moves") or payload.get("edits") or []
        if isinstance(raw_moves, list):
            for mv in raw_moves:
                if isinstance(mv, dict) and mv.get("text"):
                    mv = dict(mv)
                    mv.setdefault("source", "llm")
                    llm_moves.append(mv)
    moves = _merge_moves(llm_moves, heuristic)
    rows, applied, skipped = _apply_overlap_moves(rows, moves)
    rows = [r for r in rows if r.get("raw") or _norm_ws(str(r.get("text") or ""))]
    refined = _format_transcript_rows(rows)
    repls = list(payload.get("replacements") or [])
    if not isinstance(repls, list):
        repls = []
    refined, spell_edits = _apply_replacements(refined, repls)
    from llm_rules import refine_transcript_rules

    refined, punct_edits, _notes = refine_transcript_rules(refined, mode="safe")
    safety = payload.get("safety") or {}
    if isinstance(safety, str):
        safety = {"mode": "smart", "notes": safety}
    elif not isinstance(safety, dict):
        safety = {"mode": "smart", "notes": str(safety)}
    safety["backend"] = "llamacpp"
    safety["mode"] = "smart"
    if llm_format:
        safety["llm_format"] = llm_format
    safety["heuristic_candidates"] = heuristic
    safety["llm_splits"] = splits
    safety["invalid_splits"] = invalid_splits
    safety["skipped_moves"] = skipped
    edits = applied + spell_edits + [e for e in punct_edits if isinstance(e, dict)]
    return refined, edits, safety


def summarize_call_llamacpp(*, call_id: str, transcript_text: str) -> Dict[str, Any]:
    system = (
        "You are a Russian customer-call analyst. Return raw JSON only. "
        "Use only facts present in the transcript."
    )
    user = (
        f"call_id: {call_id}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Return one JSON object with keys: call_id, language, participants, intent, topics, "
        "timeline, entities, actions, issues_detected, quality_notes."
    )
    payload = _chat_json(
        system=system,
        user=user,
        max_tokens=640,
        schema=CallSummaryResponse,
        schema_name="call_summary",
    )
    payload.pop("_llm_format", None)
    payload.setdefault("call_id", call_id)
    payload.setdefault("language", "ru")
    payload.setdefault("participants", {"speakers": [], "roles_guess": {"agent": None, "client": None, "manager": None}})
    payload.setdefault("topics", [])
    payload.setdefault("timeline", [])
    payload.setdefault(
        "entities",
        {"companies": [], "emails": [], "phones": [], "inn": [], "dates": [], "amounts": [], "addresses": []},
    )
    payload.setdefault("actions", [])
    payload.setdefault("issues_detected", [])
    payload.setdefault("quality_notes", {"has_transfer": False, "transfer_reason": None, "asr_uncertainty": None})
    return payload


_EXTRACT_SYSTEM = (
    "Ты аналитик русских телефонных звонков. Верни только JSON по схеме. "
    "Только факты из транскрипта. Пустой список лучше, чем выдумка."
)

_EXTRACT_USER = """Извлеки четыре поля. Если факта нет — пустой список. Не дополняй нулями и не угадывай.

1) phones — только контактный телефон человека (чтобы перезвонить).
- Можно склеить диктовку по частям: «восемь» «девятьсот пятьдесят два» «ноль шестьдесят четыре»…
- digits: строго 11 цифр и первая 7 или 8, либо 10 цифр и первая 9. Никакого примера-номера в ответе быть не должно.
- Запрещено: короче 10 цифр; код объявления; госномер/регион авто; цена; номер дома; часы работы; мессенджер без номера; шаблон вроде 89000000000.
- Нет явного телефонного номера в репликах → phones: [].

2) addresses — куда приехать / где находится объект.
- Город + улица/шоссе + дом, если сказано. Можно неполный адрес, но только сказанные части.
- Не адрес: название салона без улицы, «Кривцово» как вопрос без подтверждения, госномер.
- Нет адреса → addresses: [].

3) amounts — денежные суммы.
- Цена, стоимость, выплата. value — число, currency — RUB если «рублей/₽».
- Не сумма: длительность (15–20 минут), часы работы (с 9 до 9), год авто, госномер, код.
- Нет денег → amounts: [].

4) commitments — явное обещание одному собеседнику от другого
  (приеду / заскочу / передам / перезвоню / пришлю / буду ждать в значении «договорились»).
- who_spk и to_spk — номера спикеров из транскрипта, не 0 если спикер виден.
- promise — обязательно своими словами; evidence — точная короткая цитата.
- Не обязательство: вопрос, совет, описание процесса, график работы.
- Нет обещаний → commitments: [].

call_id: {call_id}

Транскрипт:
{transcript}
"""


_PHONE_RE = re.compile(r"^(?:[78]\d{10}|9\d{9})$")


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
    plain = re.sub(r"\[\d{1,2}:\d{2}\]", " ", transcript or "")
    compact = re.sub(r"\D+", "", plain)
    body = digits[-10:] if len(digits) >= 10 else digits
    if digits in compact or body in compact:
        return True
    covered = 0
    for tok in re.findall(r"\d{2,}", plain):
        if tok in digits or tok in body:
            covered += len(tok)
    return covered >= 6


def _sanitize_extract(payload: Dict[str, Any], transcript_text: str) -> Dict[str, Any]:
    phones_out: List[dict] = []
    dropped: List[str] = []
    for item in payload.get("phones") or []:
        if not isinstance(item, dict):
            continue
        digits = _clean_phone_digits(str(item.get("digits") or ""))
        if not _is_plausible_ru_phone(digits):
            if digits:
                dropped.append(digits)
            continue
        if not _phone_grounded_in_transcript(digits, transcript_text):
            dropped.append(digits)
            continue
        item = dict(item)
        item["digits"] = digits
        phones_out.append(item)
    payload["phones"] = phones_out
    notes = str(payload.get("notes") or "").strip()
    if dropped:
        extra = "dropped_phones:" + ",".join(dropped)
        payload["notes"] = f"{notes} {extra}".strip() if notes else extra
    payload.setdefault("addresses", [])
    payload.setdefault("amounts", [])
    if not isinstance(payload["addresses"], list):
        payload["addresses"] = []
    if not isinstance(payload["amounts"], list):
        payload["amounts"] = []
    return payload


def extract_facts_llamacpp(*, call_id: str, transcript_text: str) -> Dict[str, Any]:
    payload = _chat_json(
        system=_EXTRACT_SYSTEM,
        user=_EXTRACT_USER.format(call_id=call_id, transcript=transcript_text),
        max_tokens=1024,
        schema=ExtractFactsResponse,
        schema_name="extract_facts",
    )
    fmt = payload.pop("_llm_format", "")
    payload.setdefault("call_id", call_id)
    payload.setdefault("phones", [])
    payload.setdefault("addresses", [])
    payload.setdefault("amounts", [])
    payload.setdefault("commitments", [])
    payload.setdefault("notes", "")
    payload = _sanitize_extract(payload, transcript_text)
    if fmt:
        payload["llm_format"] = fmt
    return payload


_ROLES_SYSTEM = (
    "Ты аналитик русских телефонных звонков автосалона. Верни только JSON по схеме. "
    "Роли ставь только по репликам. Не выдумывай спикеров, которых нет в тексте."
)

_ROLES_USER = """Для каждого номера «Спикер N» из транскрипта укажи роль.

Допустимые role (строго одно из четырёх):
- ivr — автоинформатор / робот / «оставайтесь на линии», «ответит первый свободный сотрудник». Нет живого диалога, нет имени человека.
- client — покупатель/продавец авто, звонит с вопросом, хочет приехать, диктует телефон/номер машины, отвечает на анкету.
- agent — сотрудник компании: приветствие от отдела/салона, перевод, оценка, продажа. Несколько агентов в одном звонке — нормально (сначала линия, потом узкий специалист).
- unknown — неясно.

Правила:
- Один спикер → одна роль. Несколько менеджеров = несколько строк role=agent с разными spk.
- title — отдел/функция, только если сказано («отдел оценки», «отдел продаж»).
- name — имя, только если спикер назвался или его назвали.
- evidence — короткая точная цитата этой роли.
- Не путай клиента и менеджера только потому, что оба представляются по имени.
- IVR почти никогда не идёт после живого диалога как «новый менеджер».

Спикеры, которые ОБЯЗАТЕЛЬНО должны быть в speakers: {speaker_ids}
call_id: {call_id}

Транскрипт:
{transcript}
"""

_ALLOWED_ROLES = {"ivr", "client", "agent", "unknown"}


def _transcript_speaker_ids(transcript_text: str) -> List[int]:
    ids = sorted({int(x) for x in re.findall(r"Спикер\s+(\d+)", transcript_text or "")})
    return ids


def _sanitize_roles(payload: Dict[str, Any], transcript_text: str) -> Dict[str, Any]:
    expected = _transcript_speaker_ids(transcript_text)
    by_spk: Dict[int, dict] = {}
    for item in payload.get("speakers") or []:
        if not isinstance(item, dict):
            continue
        try:
            spk = int(item.get("spk"))
        except (TypeError, ValueError):
            continue
        if expected and spk not in expected:
            continue
        role = str(item.get("role") or "unknown").strip().lower()
        if role not in _ALLOWED_ROLES:
            role = "unknown"
        by_spk[spk] = {
            "spk": spk,
            "role": role,
            "title": str(item.get("title") or "").strip(),
            "name": str(item.get("name") or "").strip(),
            "evidence": str(item.get("evidence") or "").strip(),
        }
    speakers = []
    for spk in expected or sorted(by_spk):
        speakers.append(
            by_spk.get(
                spk,
                {"spk": spk, "role": "unknown", "title": "", "name": "", "evidence": ""},
            )
        )
    payload["speakers"] = speakers
    return payload


def extract_roles_llamacpp(*, call_id: str, transcript_text: str) -> Dict[str, Any]:
    speaker_ids = _transcript_speaker_ids(transcript_text)
    payload = _chat_json(
        system=_ROLES_SYSTEM,
        user=_ROLES_USER.format(
            call_id=call_id,
            transcript=transcript_text,
            speaker_ids=", ".join(str(i) for i in speaker_ids) or "нет",
        ),
        max_tokens=768,
        schema=ExtractRolesResponse,
        schema_name="extract_roles",
    )
    fmt = payload.pop("_llm_format", "")
    payload.setdefault("call_id", call_id)
    payload.setdefault("speakers", [])
    payload.setdefault("notes", "")
    payload = _sanitize_roles(payload, transcript_text)
    if fmt:
        payload["llm_format"] = fmt
    return payload

