"""Energy VAD + hold-tone heuristics for transfer detection (host, no ML deps)."""

from __future__ import annotations

import array
import math
import statistics
import wave
from pathlib import Path
from typing import List, Sequence, Tuple


HoldInterval = dict  # {start, end, dur, kind: silence|music}


def _frame_rms(samples: array.array, start: int, n: int) -> float:
    if n <= 0:
        return 0.0
    acc = 0.0
    end = min(len(samples), start + n)
    for i in range(start, end):
        v = float(samples[i])
        acc += v * v
    return math.sqrt(acc / max(1, end - start))


def _dbfs(rms: float, full_scale: float = 32768.0) -> float:
    if rms <= 1e-6:
        return -100.0
    return 20.0 * math.log10(rms / full_scale)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = min(len(xs) - 1, f + 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _runs(mask: Sequence[bool], frame_sec: float) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        start = i * frame_sec
        end = j * frame_sec
        out.append((start, end, end - start))
        i = j
    return out


def _absorb_short_gaps(mask: List[bool], max_gap_frames: int) -> List[bool]:
    """Fill short False-runs between True (tiny clicks inside silence)."""
    out = list(mask)
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        left = i > 0 and out[i - 1]
        right = j < n and out[j]
        if left and right and (j - i) <= max_gap_frames:
            for k in range(i, j):
                out[k] = True
        i = j
    return out


def _merge_intervals(
    intervals: List[HoldInterval], *, min_hold_sec: float, gap_sec: float = 0.5
) -> List[HoldInterval]:
    intervals = sorted(intervals, key=lambda h: (float(h["start"]), float(h["end"])))
    merged: List[HoldInterval] = []
    for h in intervals:
        if not merged:
            merged.append(dict(h))
            continue
        prev = merged[-1]
        if float(h["start"]) <= float(prev["end"]) + gap_sec:
            prev["end"] = max(float(prev["end"]), float(h["end"]))
            prev["dur"] = float(prev["end"]) - float(prev["start"])
            if h.get("kind") == "music":
                prev["kind"] = "music"
        else:
            merged.append(dict(h))
    return [h for h in merged if float(h["dur"]) >= min_hold_sec]


def detect_hold_intervals(
    wav_path: Path,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    min_hold_sec: float = 5.0,
    absorb_blip_sec: float = 0.5,
) -> List[HoldInterval]:
    """Find mid-call deep silence and periodic hold/ringback tones.

    Two complementary detectors (kept sparse to avoid dialogue false positives):
    - silence: energy ≤ ~20th percentile (deep quiet)
    - music: regular strong beeps (period ~1.5–5.5 s) over a quiet floor
    """
    path = Path(wav_path)
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit mono PCM: {path}")
        rate = handle.getframerate() or 16000
        raw = handle.readframes(handle.getnframes())

    samples = array.array("h")
    samples.frombytes(raw)
    frame = max(1, int(rate * frame_ms / 1000.0))
    hop = max(1, int(rate * hop_ms / 1000.0))
    frame_sec = hop / float(rate)

    energies: List[float] = []
    for start in range(0, len(samples) - frame + 1, hop):
        energies.append(_frame_rms(samples, start, frame))
    if len(energies) < 10:
        return []

    dbs = [_dbfs(e) for e in energies]
    p20 = _percentile(dbs, 20)
    p60 = _percentile(dbs, 60)
    p80 = _percentile(dbs, 80)
    p95 = _percentile(dbs, 95)
    audio_dur = len(energies) * frame_sec
    max_frac = 0.45

    intervals: List[HoldInterval] = []

    # --- deep silence ---
    silence_thr = min(p20, p80 - 25.0)
    silence_mask = [d <= silence_thr for d in dbs]
    blip_frames = max(1, int(absorb_blip_sec / frame_sec))
    silence_mask = _absorb_short_gaps(silence_mask, blip_frames)
    for start, end, dur in _runs(silence_mask, frame_sec):
        if min_hold_sec <= dur <= audio_dur * max_frac:
            intervals.append(
                {"start": start, "end": end, "dur": dur, "kind": "silence"}
            )

    # --- periodic hold / ringback beeps ---
    peak_thr = max(p95 - 2.0, -18.0)
    peaks: List[float] = []
    i = 0
    n = len(dbs)
    while i < n:
        if dbs[i] < peak_thr:
            i += 1
            continue
        j = i + 1
        while j < n and dbs[j] >= peak_thr:
            j += 1
        dur_p = (j - i) * frame_sec
        if 0.12 <= dur_p <= 2.2:
            peaks.append(((i + j) / 2.0) * frame_sec)
        i = j

    if len(peaks) >= 4:
        groups: List[List[float]] = [[peaks[0]]]
        for t in peaks[1:]:
            if t - groups[-1][-1] <= 6.0:
                groups[-1].append(t)
            else:
                groups.append([t])
        for group in groups:
            if len(group) < 4:
                continue
            intervals_sec = [group[k + 1] - group[k] for k in range(len(group) - 1)]
            good = [x for x in intervals_sec if 1.5 <= x <= 5.5]
            if len(good) < 3:
                continue
            mean_i = sum(good) / len(good)
            std = statistics.pstdev(good) if len(good) > 1 else 0.0
            if mean_i <= 0 or (std / mean_i) > 0.35:
                continue
            t0 = group[0] - 0.4
            t1 = group[-1] + 0.4
            i0 = max(0, int(t0 / frame_sec))
            i1 = min(n, int(t1 / frame_sec) + 1)
            chunk = dbs[i0:i1]
            if not chunk:
                continue
            med = sorted(chunk)[len(chunk) // 2]
            if med > silence_thr + 20.0:
                continue
            loud_frac = sum(1 for d in chunk if d >= p60) / float(len(chunk))
            if loud_frac > 0.35:
                continue
            dur = t1 - t0
            if dur < max(min_hold_sec, 6.0) or dur > audio_dur * max_frac:
                continue
            intervals.append(
                {
                    "start": t0,
                    "end": t1,
                    "dur": dur,
                    "kind": "music",
                    "period": round(mean_i, 2),
                    "n_beeps": len(group),
                }
            )

    return _merge_intervals(intervals, min_hold_sec=min_hold_sec)
