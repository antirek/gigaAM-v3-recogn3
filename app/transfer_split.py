"""Detect call-transfer split points and stitch per-part diarization."""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Segment = dict  # {start, end, speaker}


# Transfer phrases. «соедин» covers соединить/соединяю/соединим.
# Avoid bare IVR «оставайтесь на линии» / «первый свободный» alone.
DEFAULT_TRANSFER_CUES = (
    r"перевед",
    r"переключ",
    r"соедин",
    # Word boundary: «передам вашему менеджеру» must NOT match.
    r"передам\s+(вас|ваш)\b",
    r"оставайтесь\s+на\s+линии",
)


def _dur(seg: Segment) -> float:
    return max(0.0, float(seg["end"]) - float(seg["start"]))


def speech_intervals(segments: Iterable[Segment]) -> List[Tuple[float, float]]:
    """Merge all speech into non-overlapping intervals (ignore speaker)."""
    pieces = sorted(
        ((float(s["start"]), float(s["end"])) for s in segments),
        key=lambda x: x[0],
    )
    if not pieces:
        return []
    merged = [list(pieces[0])]
    for start, end in pieces[1:]:
        if start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def find_gaps(
    segments: Iterable[Segment],
    *,
    min_gap: float,
    audio_end: Optional[float] = None,
) -> List[Tuple[float, float, float]]:
    """Return gaps (gap_start, gap_end, duration) between speech islands."""
    intervals = speech_intervals(segments)
    if len(intervals) < 2:
        return []
    gaps = []
    for (_, end), (start, _) in zip(intervals, intervals[1:]):
        dur = start - end
        if dur >= min_gap:
            gaps.append((end, start, dur))
    if audio_end is not None and intervals:
        # ignore trailing silence
        pass
    return gaps


def cue_times(utterances: Sequence[dict], cues: Sequence[str]) -> List[float]:
    """Return end-times of utterances matching transfer cues."""
    patterns = [re.compile(c, re.IGNORECASE) for c in cues if c.strip()]
    if not patterns:
        return []
    hits = []
    for u in utterances:
        text = str(u.get("text") or "")
        if any(p.search(text) for p in patterns):
            hits.append(float(u.get("end", u.get("start", 0.0))))
    return hits


def find_dialog_start(
    raw_segments: Sequence[Segment],
    *,
    max_ivr_end: float = 25.0,
    min_gap: float = 1.0,
    min_island: float = 2.0,
) -> float:
    """Skip lead-in IVR: start of first post-IVR speech island (≥ min_island)."""
    intervals = speech_intervals(raw_segments)
    if len(intervals) < 2:
        return 0.0
    (_a0, b0) = intervals[0]
    if b0 > max_ivr_end:
        return 0.0
    for a1, b1 in intervals[1:]:
        if (a1 - b0) < min_gap:
            continue
        if (b1 - a1) >= min_island:
            return float(a1)
        # Short blip (ring/beep): keep scanning.
        b0 = b1
    return 0.0


def find_transfer_split(
    *,
    raw_segments: Sequence[Segment],
    utterances: Sequence[dict] = (),
    min_gap: float = 8.0,
    cues: Sequence[str] = DEFAULT_TRANSFER_CUES,
    margin_sec: float = 15.0,
    audio_end: Optional[float] = None,
    audio_holds: Sequence[dict] = (),
) -> Optional[dict]:
    """Pick a split time for re-diarization around a likely transfer.

    Priority when cue present:
      1. audio hold (silence/music VAD) after cue
      2. long diar gap after cue
      3. soft diar gap after cue
      4. cue_only (cold transfer)
    Optional: long audio hold without cue (TRANSFER_HOLD_ONLY=1).
    Optional: longest diar gap without cue (TRANSFER_GAP_ONLY=1).
    """
    if audio_end is None and raw_segments:
        audio_end = max(float(s["end"]) for s in raw_segments)
    audio_end = float(audio_end or 0.0)
    gaps = find_gaps(raw_segments, min_gap=min_gap)
    soft_gaps = find_gaps(raw_segments, min_gap=1.5)
    mid_gaps = [
        g
        for g in gaps
        if g[0] >= margin_sec and (audio_end <= 0 or g[1] <= audio_end - margin_sec)
    ]

    hold_gaps: List[Tuple[float, float, float, str]] = []
    for h in audio_holds:
        start_h, end_h = float(h["start"]), float(h["end"])
        dur = float(h.get("dur") or (end_h - start_h))
        kind = str(h.get("kind") or "silence")
        if start_h < margin_sec:
            continue
        if audio_end > 0 and end_h > audio_end - margin_sec:
            continue
        if dur < min_gap * 0.6:
            continue
        hold_gaps.append((start_h, end_h, dur, kind))

    hits = cue_times(utterances, cues)
    chosen = None
    reason = ""
    split_t: Optional[float] = None
    hold_kind: Optional[str] = None
    hold_window = float(os.getenv("TRANSFER_HOLD_WINDOW_SEC", "60"))
    allow_gap_only = os.getenv("TRANSFER_GAP_ONLY", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    allow_hold_only = os.getenv("TRANSFER_HOLD_ONLY", "0").lower() in {
        "1",
        "true",
        "yes",
    }

    if hits:
        cue_t = max(hits)
        audio_after = [
            g
            for g in hold_gaps
            if g[0] >= cue_t - 1.0 and g[0] <= cue_t + hold_window
        ]
        if audio_after:
            pick = audio_after[-1]
            chosen = (pick[0], pick[1], pick[2])
            hold_kind = pick[3]
            reason = f"audio_{hold_kind}_after_cue@{cue_t:.1f}s"
        if chosen is None:
            after = [
                g
                for g in mid_gaps
                if g[0] >= cue_t - 1.0 and g[0] <= cue_t + hold_window
            ]
            if after:
                chosen = after[-1]
                reason = f"last_gap_after_cue@{cue_t:.1f}s"
        if chosen is None:
            soft_after = [
                g
                for g in soft_gaps
                if g[0] >= cue_t - 0.5 and g[0] <= cue_t + hold_window
            ]
            if soft_after:
                chosen = soft_after[0]
                reason = f"soft_gap_after_cue@{cue_t:.1f}s"
        if chosen is None:
            split_t = (
                min(cue_t + 0.4, audio_end - 5.0) if audio_end > cue_t + 10 else None
            )
            if split_t is not None and split_t > margin_sec:
                reason = f"cue_only@{cue_t:.1f}s"
                chosen = (split_t, split_t, 0.0)

    if chosen is None and allow_hold_only and hold_gaps:
        pick = max(hold_gaps, key=lambda g: g[2])
        chosen = (pick[0], pick[1], pick[2])
        hold_kind = pick[3]
        reason = f"audio_{hold_kind}_only"
    if chosen is None and allow_gap_only and mid_gaps:
        chosen = max(mid_gaps, key=lambda g: g[2])
        reason = "longest_mid_gap"
    if chosen is None:
        return None

    gap_start, gap_end, gap_dur = chosen
    if split_t is None:
        split_t = (gap_start + gap_end) / 2.0 if gap_dur > 0 else gap_start
    if split_t < 8.0 or (audio_end > 0 and audio_end - split_t < 8.0):
        return None
    dialog_start = find_dialog_start(raw_segments)
    if dialog_start >= split_t - 5.0:
        dialog_start = 0.0
    return {
        "split_t": float(split_t),
        "dialog_start": dialog_start,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "gap_dur": gap_dur,
        "reason": reason,
        "cue_hits": hits,
        "hold_kind": hold_kind,
        "audio_holds_n": len(hold_gaps),
    }


def _speaker_totals(segments: Sequence[Segment]) -> Dict[int, float]:
    totals: Dict[int, float] = {}
    for s in segments:
        spk = int(s["speaker"])
        totals[spk] = totals.get(spk, 0.0) + _dur(s)
    return totals


def _remap_local(segments: Sequence[Segment]) -> List[Segment]:
    order: List[int] = []
    for s in segments:
        spk = int(s["speaker"])
        if spk not in order:
            order.append(spk)
    mapping = {old: i for i, old in enumerate(order)}
    out = []
    for s in segments:
        item = dict(s)
        item["speaker"] = mapping[int(s["speaker"])]
        item["start"] = float(s["start"])
        item["end"] = float(s["end"])
        out.append(item)
    return out


def _first_real_tail_speaker(
    tail: Sequence[Segment],
    *,
    resume_t: float,
    min_seg_sec: float = 0.85,
) -> Optional[int]:
    """Speaker of the first solid turn after hold (skip ringback beeps)."""
    candidates = [
        s
        for s in tail
        if float(s["end"]) >= resume_t
        and _dur(s) >= min_seg_sec
        and float(s["start"]) >= resume_t - 0.35
    ]
    if not candidates:
        # Soft fallback: any long-enough segment that overlaps/after resume.
        candidates = [
            s
            for s in tail
            if float(s["end"]) >= resume_t and _dur(s) >= min_seg_sec
        ]
    if not candidates:
        return None
    first = min(candidates, key=lambda s: (float(s["start"]), float(s["end"])))
    return int(first["speaker"])


def stitch_diar_parts(
    head: Sequence[Segment],
    tail: Sequence[Segment],
    *,
    split_t: float,
    head_offset: float = 0.0,
    continuity: str = "longest",
    lead_segments: Sequence[Segment] = (),
    resume_t: Optional[float] = None,
) -> List[Segment]:
    """Combine optional lead (IVR) + head + tail diar.

    Continuity maps continuing customer in tail → longest head spk.
    With continuity=first_new, the first *real* post-hold voice is the new agent
    (hold beeps before resume_t are ignored for that decision and dropped).
    lead_segments keep their speaker ids and are prepended as-is.
    """
    head_l = _remap_local(head)
    for s in head_l:
        s["start"] = float(s["start"]) + head_offset
        s["end"] = float(s["end"]) + head_offset

    # Absolute time when post-hold dialogue resumes (end of audio hold / gap).
    resume_abs = float(resume_t) if resume_t is not None else float(split_t)

    tail_l = _remap_local(tail)
    for s in tail_l:
        s["start"] = float(s["start"]) + split_t
        s["end"] = float(s["end"]) + split_t

    # Drop hold/ringback crumbs before dialogue resumes.
    if resume_abs > split_t + 0.5:
        tail_l = [s for s in tail_l if float(s["end"]) > resume_abs - 0.15]

    lead = [dict(s) for s in lead_segments]
    lead_ids = {int(s["speaker"]) for s in lead}

    if not head_l and not lead:
        return tail_l
    if not tail_l:
        out = lead + head_l
        out.sort(key=lambda s: (float(s["start"]), float(s["end"]), int(s["speaker"])))
        return out

    # Remap head speakers to ids that do not collide with lead.
    if lead_ids and head_l:
        shift = max(lead_ids) + 1
        for s in head_l:
            s["speaker"] = int(s["speaker"]) + shift

    head_tot = _speaker_totals(head_l)
    tail_tot = _speaker_totals(tail_l)
    head_ids = sorted(head_tot, key=lambda spk: -head_tot[spk])
    tail_ids = sorted(tail_tot, key=lambda spk: -tail_tot[spk])
    used = set(lead_ids) | set(head_tot)
    next_id = (max(used) + 1) if used else 0

    # Who speaks first after hold is usually the *new* agent — not beep spikes.
    first_tail = _first_real_tail_speaker(tail_l, resume_t=resume_abs)

    tail_map: Dict[int, int] = {}
    if continuity in {"longest", "first_new"} and head_ids and tail_ids:
        if (
            continuity == "first_new"
            and first_tail is not None
            and len(tail_ids) >= 2
            and first_tail in tail_tot
        ):
            # New agent greets first; continuing customer is the other main voice.
            continuing = next((spk for spk in tail_ids if spk != first_tail), None)
            new_agent = first_tail
        else:
            # Fallback: longest on both sides = customer (often wrong if new
            # agent talks more — prefer continuity=first_new for transfers).
            continuing = tail_ids[0]
            new_agent = tail_ids[1] if len(tail_ids) > 1 else None

        if continuing is not None:
            tail_map[continuing] = head_ids[0]
        for spk in tail_ids:
            if spk in tail_map:
                continue
            while next_id in used:
                next_id += 1
            tail_map[spk] = next_id
            used.add(next_id)
            next_id += 1
        # Ensure new_agent got an id (already assigned in loop).
        _ = new_agent
    else:
        for spk in tail_ids:
            while next_id in used:
                next_id += 1
            tail_map[spk] = next_id
            used.add(next_id)
            next_id += 1

    out = lead + [dict(s) for s in head_l]
    for s in tail_l:
        item = dict(s)
        item["speaker"] = tail_map[int(s["speaker"])]
        out.append(item)
    out.sort(key=lambda s: (float(s["start"]), float(s["end"]), int(s["speaker"])))
    return out


def wav_duration_sec(path) -> float:
    import wave

    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate() or 1)
