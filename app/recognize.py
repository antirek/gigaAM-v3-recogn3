"""Host orchestrator: preprocess → diar container → merge → asr container → TXT."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

from .align import join_texts_by_utterance, pack_asr_chunks, prepare_segments
from .preprocess import cut_segment, enhance_mode_from_env, ensure_wav_16k_mono, format_ts
from .qwen_dual import run_qwen_dual
from .hold_detect import detect_hold_intervals
from .transfer_split import (
    DEFAULT_TRANSFER_CUES,
    find_transfer_split,
    stitch_diar_parts,
    wav_duration_sec,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # Prefer docker-compose v1 (this host); fall back to `docker compose` plugin.
    if shutil.which("docker-compose"):
        cmd = ["docker-compose", "-f", str(ROOT / "docker-compose.yml"), *args]
    else:
        cmd = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=check)


def _out_container_path(path: Path) -> str:
    return "/out/" + str(path.resolve().relative_to((ROOT / "out").resolve())).replace(
        "\\", "/"
    )


def _run_diar(wav: Path, diar_out: Path) -> List[dict]:
    print(f"[orch] diarization → {diar_out.name}…", flush=True)
    _compose(
        "run",
        "--rm",
        "diar",
        "--audio",
        _out_container_path(wav),
        "--out",
        _out_container_path(diar_out),
    )
    payload = json.loads(diar_out.read_text(encoding="utf-8"))
    return payload.get("segments", payload if isinstance(payload, list) else [])


def _write_transcript(out_dir: Path, transcript: Sequence[dict]) -> Path:
    lines = []
    for item in transcript:
        speaker = int(item["speaker"])
        start = float(item["start"])
        text = item["text"]
        lines.append(f"[{format_ts(start)}] Спикер {speaker}: {text}")

    transcript_json = out_dir / "transcript.json"
    transcript_txt = out_dir / "transcript.txt"
    transcript_json.write_text(
        json.dumps({"utterances": transcript}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    transcript_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return transcript_txt


def _asr_pass(wav16: Path, out_dir: Path, raw_segments: Sequence[dict]) -> List[dict]:
    merge_gap = float(os.getenv("MERGE_GAP_SEC", "1.2"))
    min_seg = float(os.getenv("MIN_SEGMENT_SEC", "0.35"))
    max_spk = int(os.getenv("MAX_SPEAKERS", "4"))
    min_spk_sec = float(os.getenv("MIN_SPK_SEC", "10"))
    min_spk_share = float(os.getenv("MIN_SPK_SHARE", "0.06"))
    max_asr = float(os.getenv("MAX_ASR_SEC", "20"))
    atomics, segments = prepare_segments(
        list(raw_segments),
        max_speakers=max_spk,
        merge_gap=merge_gap,
        min_segment=min_seg,
        min_spk_sec=min_spk_sec,
        min_spk_share=min_spk_share,
    )
    segments_path = out_dir / "segments.json"
    segments_path.write_text(
        json.dumps(
            {"segments": segments, "atomics": atomics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    asr_segments = pack_asr_chunks(segments, atomics, max_sec=max_asr)
    print(
        f"[orch] {len(raw_segments)} raw → {len(atomics)} atomics → "
        f"{len(segments)} turns → {len(asr_segments)} ASR chunks "
        f"(cut on diar pauses, max {max_asr:.0f}s)",
        flush=True,
    )

    chunks_dir = out_dir / "chunks"
    if chunks_dir.exists():
        for old in chunks_dir.glob("*.wav"):
            old.unlink()
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for i, seg in enumerate(asr_segments):
        chunk = chunks_dir / f"seg_{i:04d}_spk{seg['speaker']}.wav"
        cut_segment(wav16, chunk, float(seg["start"]), float(seg["end"]))
        manifest_items.append(
            {
                "index": i,
                "utterance_id": int(seg["utterance_id"]),
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": int(seg["speaker"]),
                "path": _out_container_path(chunk),
            }
        )

    manifest_path = out_dir / "asr_manifest.json"
    manifest_path.write_text(
        json.dumps({"segments": manifest_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    asr_out = out_dir / "asr_raw.json"

    print("[orch] ASR…", flush=True)
    _compose(
        "run",
        "--rm",
        "asr",
        "--manifest",
        _out_container_path(manifest_path),
        "--out",
        _out_container_path(asr_out),
    )

    asr_payload = json.loads(asr_out.read_text(encoding="utf-8"))
    transcript = join_texts_by_utterance(asr_payload.get("segments", []))

    # Drop obvious non-dialogue lead-in (IVR / "Сообщение.")
    if transcript:
        lead = transcript[0]["text"].strip().lower().rstrip(".!")
        if lead in {"сообщение", "гудок", "beep", "звонок"}:
            transcript = transcript[1:]
    return transcript


def _maybe_transfer_rediar(
    wav16: Path,
    out_dir: Path,
    raw_segments: List[dict],
    transcript: Sequence[dict],
) -> List[dict]:
    """If transfer cues + gap found, re-diar head/tail and stitch."""
    if not _env_bool("TRANSFER_SPLIT", True):
        return raw_segments

    gap_sec = float(os.getenv("TRANSFER_GAP_SEC", "8"))
    margin = float(os.getenv("TRANSFER_MARGIN_SEC", "15"))
    continuity = os.getenv("TRANSFER_CONTINUITY", "first_new").strip().lower()
    cues_env = os.getenv("TRANSFER_CUES", "").strip()
    cues = (
        [c.strip() for c in cues_env.split("|") if c.strip()]
        if cues_env
        else list(DEFAULT_TRANSFER_CUES)
    )
    audio_end = wav_duration_sec(wav16)
    audio_holds: List[dict] = []
    if _env_bool("TRANSFER_AUDIO_HOLD", True):
        min_hold = float(os.getenv("TRANSFER_AUDIO_HOLD_SEC", "5"))
        try:
            audio_holds = detect_hold_intervals(wav16, min_hold_sec=min_hold)
            print(
                f"[orch] audio-hold: {len(audio_holds)} intervals "
                f"(≥{min_hold:.0f}s silence/music)",
                flush=True,
            )
            for h in audio_holds[:8]:
                print(
                    f"  hold {h['kind']} {h['start']:.1f}-{h['end']:.1f} "
                    f"({h['dur']:.1f}s)",
                    flush=True,
                )
            (out_dir / "hold_detect.json").write_text(
                json.dumps({"intervals": audio_holds}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[orch] audio-hold failed: {exc}", flush=True)

    info = find_transfer_split(
        raw_segments=raw_segments,
        utterances=transcript,
        min_gap=gap_sec,
        cues=cues,
        margin_sec=margin,
        audio_end=audio_end,
        audio_holds=audio_holds,
    )
    if not info:
        stale = out_dir / "transfer_split.json"
        if stale.is_file():
            stale.unlink()
        print("[orch] transfer-split: no split point", flush=True)
        return raw_segments

    split_t = float(info["split_t"])
    dialog_start = float(info.get("dialog_start") or 0.0)
    print(
        f"[orch] transfer-split @ {split_t:.1f}s "
        f"(head from {dialog_start:.1f}s, "
        f"gap {info['gap_start']:.1f}-{info['gap_end']:.1f}, "
        f"{info['reason']})",
        flush=True,
    )

    # Keep original full-file diar for comparison.
    diar_full = out_dir / "diar.full.json"
    diar_raw = out_dir / "diar.raw.json"
    if diar_raw.is_file() and not diar_full.is_file():
        shutil.copy2(diar_raw, diar_full)

    parts_dir = out_dir / "transfer_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    head_wav = parts_dir / "head.wav"
    tail_wav = parts_dir / "tail.wav"
    # Skip IVR in the head clip — Sortformer separates agent/client better without it.
    cut_segment(wav16, head_wav, dialog_start, split_t)
    cut_segment(wav16, tail_wav, split_t, audio_end)

    head_diar = parts_dir / "diar_head.json"
    tail_diar = parts_dir / "diar_tail.json"
    head_segs = _run_diar(head_wav, head_diar)
    tail_segs = _run_diar(tail_wav, tail_diar)

    # Optional IVR island as its own speaker (ignored later if short).
    lead_segs: List[dict] = []
    if dialog_start > 1.0:
        lead_segs = [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "speaker": 0,
            }
            for s in raw_segments
            if float(s["end"]) <= dialog_start + 0.05
        ]

    # Resume after hold/gap so ringback beeps are not treated as the new agent.
    resume_t = float(info.get("gap_end") or split_t)
    stitched = stitch_diar_parts(
        head_segs,
        tail_segs,
        split_t=split_t,
        head_offset=dialog_start,
        continuity=continuity,
        lead_segments=lead_segs,
        resume_t=resume_t,
    )
    spk_ids = sorted({int(s["speaker"]) for s in stitched})
    print(
        f"[orch] transfer-split stitched: {len(stitched)} segs, speakers={spk_ids}",
        flush=True,
    )

    meta = {
        "split": info,
        "continuity": continuity,
        "speakers": spk_ids,
        "n_segments": len(stitched),
    }
    (out_dir / "transfer_split.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Docker may leave diar.raw.json as root; unlink then rewrite as host user.
    if diar_raw.is_file():
        try:
            diar_raw.unlink()
        except OSError:
            pass
    diar_raw.write_text(
        json.dumps(
            {"segments": stitched, "transfer_split": meta},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return stitched


def run_pipeline(audio: Path, out_dir: Path) -> Path:
    _load_env_file(ROOT / ".env")
    out_dir.mkdir(parents=True, exist_ok=True)

    wav16 = out_dir / "audio_16k.wav"
    mode = enhance_mode_from_env()
    print(f"[orch] preprocess → {wav16} (enhance={mode})", flush=True)
    ensure_wav_16k_mono(audio, wav16, mode=mode)

    diar_raw = out_dir / "diar.raw.json"
    skip_diar = _env_bool("SKIP_DIAR", False)
    if skip_diar and diar_raw.is_file():
        print(f"[orch] skip diar, reuse {diar_raw}", flush=True)
        raw = json.loads(diar_raw.read_text(encoding="utf-8"))
        raw_segments = raw.get("segments", raw if isinstance(raw, list) else [])
    else:
        raw_segments = _run_diar(wav16, diar_raw)

    transcript = _asr_pass(wav16, out_dir, raw_segments)
    transcript_txt = _write_transcript(out_dir, transcript)
    print(f"[orch] draft → {transcript_txt}", flush=True)

    stitched = _maybe_transfer_rediar(wav16, out_dir, raw_segments, transcript)
    if stitched is not raw_segments:
        # Soften speaker filter: first-line agent after transfer is often <10s.
        prev_sec = os.environ.get("MIN_SPK_SEC")
        prev_share = os.environ.get("MIN_SPK_SHARE")
        os.environ["MIN_SPK_SEC"] = os.getenv("TRANSFER_MIN_SPK_SEC", "4")
        os.environ["MIN_SPK_SHARE"] = os.getenv("TRANSFER_MIN_SPK_SHARE", "0.03")
        try:
            transcript = _asr_pass(wav16, out_dir, stitched)
        finally:
            if prev_sec is None:
                os.environ.pop("MIN_SPK_SEC", None)
            else:
                os.environ["MIN_SPK_SEC"] = prev_sec
            if prev_share is None:
                os.environ.pop("MIN_SPK_SHARE", None)
            else:
                os.environ["MIN_SPK_SHARE"] = prev_share
        transcript_txt = _write_transcript(out_dir, transcript)
        print(f"[orch] transfer-split done → {transcript_txt}", flush=True)
    else:
        print(f"[orch] done → {transcript_txt}", flush=True)

    # Optional secondary ASR: Qwen3-ASR (sister project) → dual_for_llm.md
    run_qwen_dual(
        audio=audio,
        out_dir=out_dir,
        wav16=wav16,
        diar_raw=diar_raw,
        gigaam_transcript_txt=transcript_txt,
    )
    return transcript_txt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize dialogue with GigaAM + Sortformer diarization"
    )
    parser.add_argument("--audio", required=True, type=Path, help="Input audio file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory under ./out (default: out/<stem>/)",
    )
    parser.add_argument(
        "--dual-qwen",
        action="store_true",
        help="Also run Qwen3-ASR (../qwen3-asr-campplus) and write dual_for_llm.md",
    )
    parser.add_argument(
        "--transfer-split",
        action="store_true",
        help="Force transfer re-diarization on (default: on)",
    )
    parser.add_argument(
        "--no-transfer-split",
        action="store_true",
        help="Disable transfer re-diarization (TRANSFER_SPLIT=0)",
    )
    args = parser.parse_args()

    if args.dual_qwen:
        os.environ["QWEN_DUAL"] = "1"
    if args.no_transfer_split:
        os.environ["TRANSFER_SPLIT"] = "0"
    elif args.transfer_split:
        os.environ["TRANSFER_SPLIT"] = "1"

    audio = args.audio
    if not audio.is_file():
        # allow path relative to project / data
        candidate = ROOT / audio
        if candidate.is_file():
            audio = candidate
        else:
            raise SystemExit(f"audio not found: {args.audio}")

    if args.out is None:
        out_dir = ROOT / "out" / audio.stem
    else:
        out_dir = args.out
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    out_resolved = out_dir.resolve()
    try:
        out_resolved.relative_to((ROOT / "out").resolve())
    except ValueError:
        raise SystemExit(f"--out must be inside {ROOT / 'out'}")

    try:
        run_pipeline(audio.resolve(), out_resolved)
    except subprocess.CalledProcessError as exc:
        print(f"command failed with code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)


if __name__ == "__main__":
    main()
