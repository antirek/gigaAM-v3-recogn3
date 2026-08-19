from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import requests


def _ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://ollama:11434").strip()


def _candidate_models() -> List[str]:
    raw = os.getenv("OLLAMA_MODEL_CANDIDATES") or os.getenv("OLLAMA_MODEL") or ""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # sensible defaults (may require adjusting to actual local tags)
    if not parts:
        parts = [
            "gemma2:9b",
            "gemma2:9b-instruct-q4_K_M",
            "gemma2:9b-instruct",
            "gemma2:9b-it",
        ]
    return parts


def _chat(model: str, *, system: str, user: str, temperature: float = 0.0) -> str:
    url = _ollama_url() + "/api/chat"
    # Default generation cap must be big enough to finish JSON.
    num_predict_default = "512"
    stop_raw = os.getenv("OLLAMA_STOP", "").strip()
    stop_list = [s for s in (x.strip() for x in stop_raw.split(",")) if s] if stop_raw else []
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Keep JSON responses short and avoid "runaway" generations.
        # Ollama supports `num_predict` and `stop` in options.
        "options": {
            "temperature": temperature,
            "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", num_predict_default)),
            **({"stop": stop_list} if stop_list else {}),
        },
        "stream": False,
    }
    timeout_s = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SEC", "120"))
    retries = int(os.getenv("OLLAMA_CHAT_RETRIES", "3"))
    backoff_s = float(os.getenv("OLLAMA_CHAT_RETRY_BACKOFF_SEC", "1.0"))

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            if r.status_code == 404 and attempt + 1 < retries:
                # Ollama can transiently return 404 during model warmup / routing.
                last_exc = RuntimeError("ollama /api/chat returned 404")
                continue
            r.raise_for_status()
            data = r.json()
            # Ollama returns {message:{content:"..."}}
            return data.get("message", {}).get("content") or ""
        except Exception as e:
            last_exc = e
            if attempt + 1 >= retries:
                break
            # Simple backoff; keep it small to reduce overall latency.
            try:
                import time

                time.sleep(backoff_s * (attempt + 1))
            except Exception:
                pass
    raise last_exc or RuntimeError("ollama chat failed")


def _try_parse_json(s: str) -> Dict[str, Any]:
    s = (s or "").strip()

    # Common Ollama pattern: ```json ... ```
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("```", "").strip()

    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Best-effort extraction: from the first '{' to the last '}'.
    # (If model followed "raw JSON only", this will be the whole string.)
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = s[start : end + 1]
        return json.loads(candidate)

    raise ValueError("failed to parse json (could not extract object)")


def _ensure_model_pulled(model: str) -> None:
    # Optional: pull model if not present. Keep safe (non-fatal on failure).
    try:
        tags = requests.get(_ollama_url() + "/api/tags", timeout=60).json().get("models") or []
        have = {t.get("name") for t in tags if isinstance(t, dict)}
        if model in have:
            return
    except Exception:
        # If /tags fails, still try pull; we'll ignore errors later.
        pass
    auto_pull = os.getenv("OLLAMA_AUTO_PULL", "1").lower() in {"1", "true", "yes", "on"}
    if not auto_pull:
        return
    try:
        requests.post(_ollama_url() + "/api/pull", json={"name": model}, timeout=60).raise_for_status()
    except Exception:
        # Pull may take long; we ignore here and rely on chat failure fallback.
        pass


def refine_transcript_ollama(transcript_text: str, *, mode: str = "safe") -> Tuple[str, List[dict], Dict[str, Any]]:
    system = (
        "You are a Russian call-transcription editor. Output MUST be valid JSON and nothing else. "
        "Do not invent facts, entities, names, numbers. Preserve speaker ids and timestamps exactly. "
        "Do NOT wrap output into ``` or any markdown fences; return raw JSON only."
    )
    safe_rules = (
        "SAFE mode: only fix punctuation/whitespace, normalize filler like 'э-э-э'->'э-э', "
        "and fix obvious typos without changing words/numbers content. Keep semantics."
    )
    smart_rules = (
        "SMART mode: you may also correct likely ASR errors, but still never invent new entities; "
        "if unsure about a token (names/numbers) keep original. "
        "If overlapping speech is stuck in the current speaker's line, cut that fragment "
        "and prepend it to the NEXT existing line of the other speaker. "
        "Do not create new timestamps. If unsure, leave as-is."
    )
    rules = safe_rules if mode == "safe" else smart_rules

    user = (
        f"{rules}\n\n"
        "Input transcript (multiple lines):\n"
        f"{transcript_text}\n\n"
        "Return JSON with keys: refined_transcript (string), edits (array of objects with type+count), "
        "safety {mode, notes}. "
        "refined_transcript must keep the same format '[timestamp] Спикер N: text' per line."
    )

    last_err: Exception | None = None
    for model in _candidate_models():
        _ensure_model_pulled(model)
        try:
            content = _chat(model, system=system, user=user, temperature=0.0)
            payload = _try_parse_json(content)
            refined = payload.get("refined_transcript") or ""
            edits = payload.get("edits") or []
            safety = payload.get("safety") or {}
            if not refined:
                raise ValueError("empty refined_transcript")
            return refined + ("" if refined.endswith("\n") else "\n"), edits, safety
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("ollama refine failed")


def summarize_call_ollama(*, call_id: str, transcript_text: str) -> Dict[str, Any]:
    system = (
        "You are a Russian customer-call analyst. Output MUST be valid JSON and nothing else. "
        "Do not invent facts. Use only what's present in the transcript. "
        "Do NOT wrap output into ``` or any markdown fences; return raw JSON only."
    )
    user = (
        f"call_id: {call_id}\n\n"
        "Transcript:\n"
        f"{transcript_text}\n\n"
        "Return JSON with schema keys:\n"
        "- call_id, language\n"
        "- participants: {speakers:[ints], roles_guess:{agent:int|null, client:int|null, manager:int|null}}\n"
        "- intent (string)\n"
        "- topics (array strings)\n"
        "- timeline (array of {t_hint,start_hint,event})\n"
        "- entities {companies, emails, phones, inn, dates, amounts, addresses}\n"
        "- actions (array {who,action,deadline})\n"
        "- issues_detected (array {issue,evidence,severity})\n"
        "- quality_notes {has_transfer:boolean, transfer_reason, asr_uncertainty}\n\n"
        "Be concise but factual."
    )

    last_err: Exception | None = None
    for model in _candidate_models():
        _ensure_model_pulled(model)
        try:
            content = _chat(model, system=system, user=user, temperature=0.0)
            payload = _try_parse_json(content)
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
            payload.setdefault(
                "quality_notes",
                {"has_transfer": False, "transfer_reason": None, "asr_uncertainty": None},
            )
            return payload
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("ollama summarize failed")

