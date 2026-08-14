"""ASR CLI: transcribe WAV chunks with GigaAM v3_e2e_rnnt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def run(manifest: Path, out: Path) -> None:
    import gigaam
    import torch

    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
    cache = os.getenv("GIGAAM_CACHE", "/data/models/gigaam")
    Path(cache).mkdir(parents=True, exist_ok=True)

    print(
        f"[asr] device={device} torch={torch.__version__} model={model_name}",
        flush=True,
    )
    if device == "cuda" and torch.cuda.is_available():
        print(f"[asr] gpu={torch.cuda.get_device_name(0)}", flush=True)

    os.environ.setdefault("GIGAAM_CACHE", cache)
    model = gigaam.load_model(
        model_name,
        device=device if device in {"cpu", "cuda"} else "cuda",
        fp16_encoder=False,
        use_flash=False,
        download_root=cache,
    )

    items = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(items, dict) and "segments" in items:
        items = items["segments"]

    results = []
    for i, item in enumerate(items):
        wav = Path(item["path"])
        if not wav.is_file():
            print(f"[asr] missing {wav}", file=sys.stderr, flush=True)
            results.append({**item, "text": "", "error": "missing_file"})
            continue
        print(f"[asr] ({i+1}/{len(items)}) {wav.name}", flush=True)
        try:
            result = model.transcribe(str(wav))
            if hasattr(result, "text"):
                text = result.text
            elif isinstance(result, dict):
                text = result.get("text") or result.get("transcription") or str(result)
            else:
                text = str(result)
            text = (text or "").strip()
        except Exception as exc:
            print(f"[asr] fail {wav}: {exc}", file=sys.stderr, flush=True)
            results.append({**item, "text": "", "error": str(exc)})
            continue
        results.append({**item, "text": text})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"segments": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[asr] wrote {out} ({len(results)} utterances)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="GigaAM batch ASR from manifest")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="JSON list of {path, start, end, speaker} or {segments:[...]}",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    run(args.manifest, args.out)


if __name__ == "__main__":
    main()
