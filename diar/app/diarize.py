"""Diarization CLI: WAV → JSON segments via NVIDIA Sortformer (NeMo)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _configure_streaming(model) -> None:
    mods = model.sortformer_modules
    mods.chunk_len = int(os.getenv("DIAR_CHUNK_LEN", "340"))
    mods.chunk_right_context = int(os.getenv("DIAR_RIGHT_CONTEXT", "40"))
    mods.fifo_len = int(os.getenv("DIAR_FIFO_LEN", "40"))
    mods.spkcache_update_period = int(os.getenv("DIAR_UPDATE_PERIOD", "300"))
    mods.spkcache_len = int(os.getenv("DIAR_SPKCACHE_LEN", "188"))
    if hasattr(mods, "_check_streaming_parameters"):
        mods._check_streaming_parameters()


def _speaker_id(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits == "":
        raise ValueError(f"no speaker id in {value!r}")
    return int(digits)


def _parse_segment(item) -> dict:
    """Normalize NeMo diarize() item to {start, end, speaker} (0-based)."""
    if isinstance(item, dict):
        start = float(item.get("start", item.get("begin", item.get("start_time"))))
        end = float(item.get("end", item.get("end_time")))
        spk = item.get("speaker", item.get("label", item.get("speaker_index")))
        return {"start": start, "end": end, "speaker": _speaker_id(spk)}

    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {
            "start": float(item[0]),
            "end": float(item[1]),
            "speaker": _speaker_id(item[2]),
        }

    if isinstance(item, str):
        parts = item.replace(",", " ").split()
        if len(parts) >= 3:
            return {
                "start": float(parts[0]),
                "end": float(parts[1]),
                "speaker": _speaker_id(parts[2]),
            }

    raise ValueError(f"Unrecognized segment format: {item!r}")


def run(audio: Path, out: Path) -> None:
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diar] device={device} torch={torch.__version__}", flush=True)
    if device == "cuda":
        print(f"[diar] gpu={torch.cuda.get_device_name(0)}", flush=True)

    model_name = os.getenv(
        "DIAR_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2.1"
    )
    local_nemo = os.getenv("DIAR_NEMO_PATH", "").strip()

    if local_nemo and Path(local_nemo).is_file():
        print(f"[diar] restore_from {local_nemo}", flush=True)
        model = SortformerEncLabelModel.restore_from(
            restore_path=local_nemo, map_location=device, strict=False
        )
    else:
        print(f"[diar] from_pretrained {model_name}", flush=True)
        model = SortformerEncLabelModel.from_pretrained(model_name)

    if device == "cuda":
        model = model.cuda()
    model.eval()
    _configure_streaming(model)

    print(f"[diar] diarize {audio}", flush=True)
    predicted = model.diarize(audio=[str(audio)], batch_size=1)
    raw = predicted[0] if predicted else []
    segments = []
    for item in raw:
        try:
            segments.append(_parse_segment(item))
        except Exception as exc:
            print(f"[diar] skip segment {item!r}: {exc}", file=sys.stderr, flush=True)

    segments.sort(key=lambda s: (s["start"], s["end"], s["speaker"]))
    payload = {
        "audio": str(audio),
        "model": model_name if not local_nemo else local_nemo,
        "device": device,
        "segments": segments,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diar] wrote {out} ({len(segments)} segments)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sortformer diarization → JSON")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"audio not found: {args.audio}")
    run(args.audio, args.out)


if __name__ == "__main__":
    main()
