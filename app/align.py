"""Segment postprocess: keep 2 speakers, resolve overlaps, merge gaps."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple


Segment = dict  # {start: float, end: float, speaker: int}


def _duration(seg: Segment) -> float:
    return max(0.0, float(seg["end"]) - float(seg["start"]))


def keep_top_speakers(segments: list[Segment], max_speakers: int = 2) -> list[Segment]:
    totals: dict[int, float] = {}
    for seg in segments:
        spk = int(seg["speaker"])
        totals[spk] = totals.get(spk, 0.0) + _duration(seg)
    if len(totals) <= max_speakers:
        return segments
    keep = {
        spk
        for spk, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[
            :max_speakers
        ]
    }
    return [s for s in segments if int(s["speaker"]) in keep]


def remap_speakers(segments: list[Segment]) -> list[Segment]:
    """Map speaker ids to 1..N by first appearance order."""
    order: list[int] = []
    for seg in segments:
        spk = int(seg["speaker"])
        if spk not in order:
            order.append(spk)
    mapping = {old: i + 1 for i, old in enumerate(order)}
    out = []
    for seg in segments:
        item = dict(seg)
        item["speaker"] = mapping[int(seg["speaker"])]
        out.append(item)
    return out


def resolve_overlaps(segments: list[Segment]) -> list[Segment]:
    """Make timeline non-overlapping; dominant = longer total speech wins ties by start."""
    if not segments:
        return []

    totals: dict[int, float] = {}
    for seg in segments:
        spk = int(seg["speaker"])
        totals[spk] = totals.get(spk, 0.0) + _duration(seg)

    sorted_segs = sorted(
        segments,
        key=lambda s: (float(s["start"]), -_duration(s), int(s["speaker"])),
    )
    result: list[Segment] = []
    for seg in sorted_segs:
        start = float(seg["start"])
        end = float(seg["end"])
        spk = int(seg["speaker"])
        if end <= start:
            continue
        if not result:
            result.append({"start": start, "end": end, "speaker": spk})
            continue

        prev = result[-1]
        if start >= prev["end"]:
            result.append({"start": start, "end": end, "speaker": spk})
            continue

        # Overlap with previous
        if spk == prev["speaker"]:
            prev["end"] = max(prev["end"], end)
            continue

        overlap_start = start
        # Prefer speaker with more total duration; tie → keep previous (earlier arrival)
        if totals.get(spk, 0.0) > totals.get(prev["speaker"], 0.0):
            if overlap_start > prev["start"]:
                prev["end"] = overlap_start
            else:
                result.pop()
            if end > overlap_start:
                result.append({"start": overlap_start, "end": end, "speaker": spk})
        else:
            if end > prev["end"]:
                result.append({"start": prev["end"], "end": end, "speaker": spk})
        result = [s for s in result if s["end"] > s["start"] + 1e-4]

    return result


def merge_same_speaker(segments: list[Segment], gap: float = 0.5) -> list[Segment]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            int(seg["speaker"]) == int(prev["speaker"])
            and float(seg["start"]) - float(prev["end"]) <= gap
        ):
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
        else:
            merged.append(dict(seg))
    return merged


def absorb_short_interruptions(
    segments: list[Segment],
    max_frag: float = 0.7,
    max_gap: float = 0.45,
) -> list[Segment]:
    """Reattach a short opposite-speaker blip to surrounding speaker."""
    if len(segments) < 2:
        return segments
    out = [dict(segments[0])]
    i = 1
    while i < len(segments):
        cur = dict(segments[i])
        prev = out[-1]
        dur = _duration(cur)
        gap = float(cur["start"]) - float(prev["end"])
        if (
            int(cur["speaker"]) != int(prev["speaker"])
            and dur <= max_frag
            and 0 <= gap <= max_gap
        ):
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            if nxt is not None and int(nxt["speaker"]) == int(prev["speaker"]):
                prev["end"] = max(float(prev["end"]), float(cur["end"]))
                i += 1
                continue
            if _duration(prev) >= 1.0:
                prev["end"] = max(float(prev["end"]), float(cur["end"]))
                i += 1
                continue
        out.append(cur)
        i += 1
    return out


def drop_short(segments: list[Segment], min_sec: float = 0.35) -> list[Segment]:
    return [s for s in segments if _duration(s) >= min_sec]


def _hard_split(
    start: float, end: float, speaker: int, utterance_id: int, max_sec: float
) -> list[Segment]:
    """Last-resort fixed-size split when a single pause-free span is still too long."""
    out: list[Segment] = []
    t = start
    while t < end - 1e-4:
        chunk_end = min(end, t + max_sec)
        out.append(
            {
                "start": t,
                "end": chunk_end,
                "speaker": speaker,
                "utterance_id": utterance_id,
            }
        )
        t = chunk_end
    return out


def pack_asr_chunks(
    turns: list[Segment],
    atomics: list[Segment],
    max_sec: float = 20.0,
) -> list[Segment]:
    """Build ASR chunks ≤ max_sec, cutting on diar pause boundaries inside a turn.

    Atomics are same-speaker spans before merge (natural pauses). Turns are
    merged utterances for the final transcript (utterance_id).
    """
    if max_sec <= 0:
        return [
            {
                "start": float(t["start"]),
                "end": float(t["end"]),
                "speaker": int(t["speaker"]),
                "utterance_id": i,
            }
            for i, t in enumerate(turns)
        ]

    out: list[Segment] = []
    for uid, turn in enumerate(turns):
        t0 = float(turn["start"])
        t1 = float(turn["end"])
        spk = int(turn["speaker"])
        pieces = [
            a
            for a in atomics
            if int(a["speaker"]) == spk
            and float(a["start"]) >= t0 - 1e-3
            and float(a["end"]) <= t1 + 1e-3
        ]
        if not pieces:
            out.extend(_hard_split(t0, t1, spk, uid, max_sec))
            continue

        chunk_start: Optional[float] = None
        chunk_end: Optional[float] = None
        for piece in pieces:
            p0 = float(piece["start"])
            p1 = float(piece["end"])
            if chunk_start is None:
                # piece alone may already exceed max_sec
                if p1 - p0 > max_sec:
                    out.extend(_hard_split(p0, p1, spk, uid, max_sec))
                    chunk_start = chunk_end = None
                else:
                    chunk_start, chunk_end = p0, p1
                continue

            assert chunk_end is not None
            if p1 - chunk_start <= max_sec:
                chunk_end = p1
            else:
                out.extend(_hard_split(chunk_start, chunk_end, spk, uid, max_sec))
                if p1 - p0 > max_sec:
                    out.extend(_hard_split(p0, p1, spk, uid, max_sec))
                    chunk_start = chunk_end = None
                else:
                    chunk_start, chunk_end = p0, p1

        if chunk_start is not None and chunk_end is not None:
            out.extend(_hard_split(chunk_start, chunk_end, spk, uid, max_sec))
    return out


# Back-compat alias
def split_long_segments(
    segments: list[Segment], max_sec: float = 20.0
) -> list[Segment]:
    return pack_asr_chunks(segments, segments, max_sec=max_sec)


def join_texts_by_utterance(items: list[dict]) -> list[dict]:
    """Merge ASR chunk results that share utterance_id into one utterance."""
    if not items:
        return []
    grouped: dict[int, dict] = {}
    order: list[int] = []
    for item in items:
        text = (item.get("text") or "").strip()
        uid = item.get("utterance_id")
        if uid is None:
            uid = item.get("index", id(item))
        uid = int(uid)
        if uid not in grouped:
            grouped[uid] = {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "speaker": int(item["speaker"]),
                "text": text,
            }
            order.append(uid)
        else:
            g = grouped[uid]
            g["end"] = max(g["end"], float(item["end"]))
            g["start"] = min(g["start"], float(item["start"]))
            if text:
                g["text"] = f"{g['text']} {text}".strip() if g["text"] else text
    return [grouped[uid] for uid in order if grouped[uid].get("text")]


def prepare_segments(
    segments: Iterable[Segment],
    *,
    max_speakers: int = 2,
    merge_gap: float = 0.5,
    min_segment: float = 0.35,
) -> Tuple[List[Segment], List[Segment]]:
    """Return (atomics, turns).

    atomics — spans after overlap/absorb (pause boundaries for ASR cuts)
    turns — merged same-speaker utterances for the final transcript
    """
    segs = [dict(s) for s in segments]
    for s in segs:
        s["start"] = float(s["start"])
        s["end"] = float(s["end"])
        s["speaker"] = int(s["speaker"])
    segs = keep_top_speakers(segs, max_speakers=max_speakers)
    segs = resolve_overlaps(segs)
    segs = absorb_short_interruptions(segs)
    segs = drop_short(segs, min_sec=min_segment)
    segs = remap_speakers(segs)
    atomics = [dict(s) for s in segs]
    turns = merge_same_speaker(atomics, gap=merge_gap)
    return atomics, turns


def postprocess(
    segments: Iterable[Segment],
    *,
    max_speakers: int = 2,
    merge_gap: float = 0.5,
    min_segment: float = 0.35,
) -> list[Segment]:
    _, turns = prepare_segments(
        segments,
        max_speakers=max_speakers,
        merge_gap=merge_gap,
        min_segment=min_segment,
    )
    return turns
