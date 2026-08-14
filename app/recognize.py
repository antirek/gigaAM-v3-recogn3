"""Host orchestrator: preprocess → diar container → merge → asr container → TXT."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .align import join_texts_by_utterance, pack_asr_chunks, prepare_segments
from .preprocess import cut_segment, enhance_mode_from_env, ensure_wav_16k_mono, format_ts

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


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # Prefer docker-compose v1 (this host); fall back to `docker compose` plugin.
    if shutil.which("docker-compose"):
        cmd = ["docker-compose", "-f", str(ROOT / "docker-compose.yml"), *args]
    else:
        cmd = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=check)


def _rel_to_root(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        # Outside project: must live under ./data or ./out via bind mounts
        raise SystemExit(
            f"Path must be inside project root ({ROOT}): {path}"
        )


def run_pipeline(audio: Path, out_dir: Path) -> Path:
    _load_env_file(ROOT / ".env")
    out_dir.mkdir(parents=True, exist_ok=True)

    wav16 = out_dir / "audio_16k.wav"
    mode = enhance_mode_from_env()
    print(f"[orch] preprocess → {wav16} (enhance={mode})", flush=True)
    ensure_wav_16k_mono(audio, wav16, mode=mode)

    diar_raw = out_dir / "diar.raw.json"
    # Paths inside containers: /data and /out are mounted from ./data and ./out
    # Put working files under out/ so both services see them at /out/...
    wav_in_container = "/out/" + str(wav16.relative_to(ROOT / "out")).replace("\\", "/")
    diar_out_container = "/out/" + str(diar_raw.relative_to(ROOT / "out")).replace(
        "\\", "/"
    )

    # Ensure audio is under out/ (already is)
    skip_diar = os.getenv("SKIP_DIAR", "").lower() in {"1", "true", "yes"}
    if skip_diar and diar_raw.is_file():
        print(f"[orch] skip diar, reuse {diar_raw}", flush=True)
    else:
        print("[orch] diarization…", flush=True)
        _compose(
            "run",
            "--rm",
            "diar",
            "--audio",
            wav_in_container,
            "--out",
            diar_out_container,
        )

    raw = json.loads(diar_raw.read_text(encoding="utf-8"))
    raw_segments = raw.get("segments", raw if isinstance(raw, list) else [])
    merge_gap = float(os.getenv("MERGE_GAP_SEC", "1.2"))
    min_seg = float(os.getenv("MIN_SEGMENT_SEC", "0.35"))
    max_spk = int(os.getenv("MAX_SPEAKERS", "2"))
    max_asr = float(os.getenv("MAX_ASR_SEC", "20"))
    atomics, segments = prepare_segments(
        raw_segments,
        max_speakers=max_spk,
        merge_gap=merge_gap,
        min_segment=min_seg,
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
                "path": "/out/"
                + str(chunk.relative_to(ROOT / "out")).replace("\\", "/"),
            }
        )

    manifest_path = out_dir / "asr_manifest.json"
    manifest_path.write_text(
        json.dumps({"segments": manifest_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    asr_out = out_dir / "asr_raw.json"
    manifest_c = "/out/" + str(manifest_path.relative_to(ROOT / "out")).replace(
        "\\", "/"
    )
    asr_out_c = "/out/" + str(asr_out.relative_to(ROOT / "out")).replace("\\", "/")

    print("[orch] ASR…", flush=True)
    _compose(
        "run",
        "--rm",
        "asr",
        "--manifest",
        manifest_c,
        "--out",
        asr_out_c,
    )

    asr_payload = json.loads(asr_out.read_text(encoding="utf-8"))
    asr_segments = asr_payload.get("segments", [])
    transcript = join_texts_by_utterance(asr_segments)

    # Drop obvious non-dialogue lead-in (IVR / "Сообщение.")
    if transcript:
        lead = transcript[0]["text"].strip().lower().rstrip(".!")
        if lead in {"сообщение", "гудок", "beep", "звонок"}:
            transcript = transcript[1:]

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
    print(f"[orch] done → {transcript_txt}", flush=True)
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
    args = parser.parse_args()

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
