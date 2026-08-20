#!/usr/bin/env python3
"""
Local LLM runner (MVP).

This repo currently doesn't run Gemma4 weights. For the first iteration we
implement a deterministic "rules" backend so:
  - transcript.refined.txt is produced
  - call_summary.json is produced
  - batch_summary.json is produced

Later you can swap backend to Ollama/vLLM/Gemma4 by extending llm_backend.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from llm_natasha import extract_facts_natasha
from llm_hybrid import extract_facts_hybrid


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_refine(args: argparse.Namespace) -> int:
    from llm_backend import refine_transcript

    inp = Path(args.input)
    out = Path(args.output)
    text = _read_text(inp)
    refined, edits, notes = refine_transcript(text, mode=args.mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(refined, encoding="utf-8")
    _write_json(
        Path(args.debug_output),
        {"mode": args.mode, "notes": notes, "edits": edits},
    )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from llm_backend import extract_facts

    inp = Path(args.input)
    out = Path(args.output)
    call_id = args.call_id or inp.parent.name
    payload = extract_facts(call_id=call_id, transcript_text=_read_text(inp))
    payload["call_id"] = call_id
    _write_json(out, payload)
    return 0


def cmd_extract_natasha(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    out = Path(args.output)
    call_id = args.call_id or inp.parent.name
    payload = extract_facts_natasha(call_id=call_id, transcript_text=_read_text(inp))
    payload["call_id"] = call_id
    _write_json(out, payload)
    return 0


def cmd_extract_hybrid(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    out = Path(args.output)
    call_id = args.call_id or inp.parent.name
    payload = extract_facts_hybrid(call_id=call_id, transcript_text=_read_text(inp))
    payload["call_id"] = call_id
    _write_json(out, payload)
    return 0


def cmd_roles(args: argparse.Namespace) -> int:
    from llm_backend import extract_roles

    inp = Path(args.input)
    out = Path(args.output)
    call_id = args.call_id or inp.parent.name
    payload = extract_roles(call_id=call_id, transcript_text=_read_text(inp))
    payload["call_id"] = call_id
    _write_json(out, payload)
    return 0


def cmd_call_summarize(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from llm_backend import summarize_call

    refined_text = _read_text(inp)
    min_chars = int(os.getenv("SUMMARIZE_MIN_TEXT_CHARS", "30"))
    if len((refined_text or "").strip()) < min_chars:
        # Avoid hallucinating summaries for empty/near-empty transcripts.
        # Keep output schema stable: empty lists + narrative fields empty.
        empty = {
            "call_id": out_dir.name,
            "language": "ru",
            "participants": {
                "speakers": [],
                "roles_guess": {"agent": None, "client": None, "manager": None},
            },
            "intent": "",
            "topics": [],
            "timeline": [],
            "entities": {
                "companies": [],
                "emails": [],
                "phones": [],
                "inn": [],
                "dates": [],
                "amounts": [],
                "addresses": [],
            },
            "actions": [],
            "issues_detected": [],
            "quality_notes": {
                "has_transfer": False,
                "transfer_reason": None,
                "asr_uncertainty": "empty_transcript_skipped",
            },
        }
        _write_json(out_dir / "call_summary.json", empty)
        (out_dir / "call_summary.md").write_text(_render_call_summary_md(empty), encoding="utf-8")
        return 0
    # call_id from parent dir name: out/<tag>/<stem>
    call_id = out_dir.name
    summary = summarize_call(call_id=call_id, transcript_text=refined_text)

    _write_json(out_dir / "call_summary.json", summary)
    # also a human-readable md
    md_path = out_dir / "call_summary.md"
    md_path.write_text(_render_call_summary_md(summary), encoding="utf-8")
    return 0


def cmd_batch_summarize(args: argparse.Namespace) -> int:
    from llm_backend import summarize_batch

    out_dir = Path(args.out_dir)
    base_dir = Path(args.input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expect per-call `call_summary.json` inside base_dir/*/
    summaries = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        p = d / "call_summary.json"
        if not p.is_file():
            continue
        try:
            summaries.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass

    payload = summarize_batch(date_hint=args.date, summaries=summaries)
    _write_json(out_dir / "batch_summary.json", payload)
    (out_dir / "batch_summary.md").write_text(_render_batch_summary_md(payload), encoding="utf-8")
    return 0


def _render_call_summary_md(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Call summary: {summary.get('call_id')}")
    lines.append("")
    lines.append(f"**Intent:** {summary.get('intent')}")
    lines.append("")
    lines.append("## Topics")
    for t in summary.get("topics") or []:
        lines.append(f"- {t}")
    lines.append("")
    lines.append("## Issues detected")
    for it in summary.get("issues_detected") or []:
        sev = it.get("severity") or "med"
        issue = it.get("issue") or ""
        evidence = (it.get("evidence") or "").strip()
        if evidence:
            lines.append(f"- ({sev}) {issue}: {evidence}")
        else:
            lines.append(f"- ({sev}) {issue}")
    lines.append("")
    lines.append("## Actions")
    for a in summary.get("actions") or []:
        who = (a.get("who") or "").strip()
        action = (a.get("action") or "").strip()
        deadline = a.get("deadline")
        if deadline is None or (isinstance(deadline, str) and not deadline.strip()):
            deadline_txt = "no deadline"
        else:
            deadline_txt = str(deadline)

        prefix = f"{who}: " if who else ""
        lines.append(f"- {prefix}{action} ({deadline_txt})")
    return "\n".join(lines).rstrip() + "\n"


def _render_batch_summary_md(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Сводный отчёт за день")
    lines.append("")
    lines.append(f"**Дата:** {payload.get('date')}")
    lines.append(f"**Звонков:** {payload.get('n_calls')}")
    if payload.get("n_calls_total") and payload.get("n_calls_total") != payload.get("n_calls"):
        lines.append(f"**Всего в папке (с пустыми):** {payload.get('n_calls_total')}")
    backend = payload.get("backend")
    if backend:
        mode = payload.get("mode")
        n_chunks = payload.get("n_chunks")
        extra = f" ({mode}, chunks={n_chunks})" if mode else ""
        lines.append(f"**Backend:** {backend}{extra}")
    lines.append("")

    exec_summary = (payload.get("executive_summary") or "").strip()
    if exec_summary:
        lines.append("## Общая картина дня")
        lines.append("")
        lines.append(exec_summary)
        lines.append("")

    key_moments = payload.get("key_moments") or []
    if key_moments:
        lines.append("## Ключевые моменты")
        lines.append("")
        for item in key_moments:
            lines.append(f"- {item}")
        lines.append("")

    recurring = payload.get("recurring_problems") or []
    if recurring:
        lines.append("## Повторяющиеся проблемы")
        lines.append("")
        for it in recurring:
            if not isinstance(it, dict):
                continue
            text = it.get("text") or ""
            count = it.get("count") or len(it.get("calls") or [])
            calls = it.get("calls") or []
            suffix = f" ({count} звонков"
            if calls:
                suffix += f": {', '.join(str(c) for c in calls[:6])}"
                if len(calls) > 6:
                    suffix += "…"
            suffix += ")"
            lines.append(f"- {text}{suffix}")
        lines.append("")

    positive = payload.get("positive_moments") or []
    if positive:
        lines.append("## Положительные моменты")
        lines.append("")
        for it in positive:
            if not isinstance(it, dict):
                continue
            text = it.get("text") or ""
            calls = it.get("calls") or []
            if calls:
                lines.append(f"- {text} (звонки: {', '.join(str(c) for c in calls[:6])})")
            else:
                lines.append(f"- {text}")
        lines.append("")

    risks = payload.get("potential_risks") or []
    if risks:
        lines.append("## Потенциальные риски")
        lines.append("")
        for it in risks:
            if not isinstance(it, dict):
                continue
            sev = it.get("severity") or "medium"
            risk = it.get("risk") or ""
            evidence = (it.get("evidence") or "").strip()
            calls = it.get("calls") or []
            line = f"- ({sev}) {risk}"
            if evidence:
                line += f": {evidence}"
            if calls:
                line += f" [звонки: {', '.join(str(c) for c in calls[:6])}]"
            lines.append(line)
        lines.append("")

    topics = payload.get("top_topics") or []
    if topics:
        lines.append("## Основные темы")
        lines.append("")
        for t in topics:
            lines.append(f"- {t}")
        lines.append("")

    recs = payload.get("recommendations") or []
    if recs:
        lines.append("## Рекомендации")
        lines.append("")
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")

    overall = payload.get("overall") or {}
    if overall.get("top_intents") or overall.get("top_topics") or overall.get("top_issues"):
        lines.append("## Статистика (rules fallback)")
        lines.append("")
        if overall.get("top_intents"):
            lines.append("### Top intents")
            for it in overall["top_intents"]:
                lines.append(f"- {it['intent']}: {it['count']}")
            lines.append("")
        if overall.get("top_topics"):
            lines.append("### Top topics")
            for it in overall["top_topics"]:
                lines.append(f"- {it['topic']}: {it['count']}")
            lines.append("")
        if overall.get("top_issues"):
            lines.append("### Top issues")
            for it in overall["top_issues"]:
                lines.append(f"- {it['issue']}: {it['count']}")
            lines.append("")

    clusters = payload.get("clusters") or []
    if clusters:
        lines.append("## Clusters")
        for c in clusters:
            lines.append(f"- {c.get('cluster_name')}: {c.get('description')}")
            calls = c.get("calls") or []
            if calls:
                lines.append(f"  - calls: {', '.join(calls[:10])}{'...' if len(calls)>10 else ''}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Local LLM MVP (rules backend).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ref = sub.add_parser("refine", help="Improve transcript text")
    p_ref.add_argument("--input", required=True, type=str)
    p_ref.add_argument("--output", required=True, type=str)
    p_ref.add_argument("--debug-output", required=True, type=str)
    p_ref.add_argument("--mode", choices=["safe", "smart"], default="safe")
    p_ref.set_defaults(fn=cmd_refine)

    p_ext = sub.add_parser("extract", help="Extract phones and commitments")
    p_ext.add_argument("--input", required=True, type=str)
    p_ext.add_argument("--output", required=True, type=str)
    p_ext.add_argument("--call-id", default="", type=str)
    p_ext.set_defaults(fn=cmd_extract)

    p_nat = sub.add_parser("extract-natasha", help="Extract phones/addresses/amounts without LLM")
    p_nat.add_argument("--input", required=True, type=str)
    p_nat.add_argument("--output", required=True, type=str)
    p_nat.add_argument("--call-id", default="", type=str)
    p_nat.set_defaults(fn=cmd_extract_natasha)

    p_hyb = sub.add_parser("extract-hybrid", help="Natasha phones/addr/money + GLiNER people/orgs/cars")
    p_hyb.add_argument("--input", required=True, type=str)
    p_hyb.add_argument("--output", required=True, type=str)
    p_hyb.add_argument("--call-id", default="", type=str)
    p_hyb.set_defaults(fn=cmd_extract_hybrid)

    p_roles = sub.add_parser("roles", help="Label speaker roles (ivr/client/agent)")
    p_roles.add_argument("--input", required=True, type=str)
    p_roles.add_argument("--output", required=True, type=str)
    p_roles.add_argument("--call-id", default="", type=str)
    p_roles.set_defaults(fn=cmd_roles)

    p_call = sub.add_parser("summarize-call", help="Summarize one call")
    p_call.add_argument("--input", required=True, type=str, help="refined transcript file")
    p_call.add_argument("--out-dir", required=True, type=str)
    p_call.set_defaults(fn=cmd_call_summarize)

    p_batch = sub.add_parser("summarize-batch", help="Summarize a batch folder")
    p_batch.add_argument("--input-dir", required=True, type=str)
    p_batch.add_argument("--out-dir", required=True, type=str)
    p_batch.add_argument("--date", default="")
    p_batch.set_defaults(fn=cmd_batch_summarize)

    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

