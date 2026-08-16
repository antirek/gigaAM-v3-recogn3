"""Optional second ASR pass: Qwen3-ASR (sister project image) + Sortformer merge.

Produces transcript_qwen.txt and dual_for_llm.md for later LLM compilation.
Does not replace the primary GigaAM transcript.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]


def _truthy(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def qwen_project_root() -> Path:
    raw = os.getenv("QWEN_PROJECT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT.parent / "qwen3-asr-campplus").resolve()


def _load_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compose_qwen(*args: str, qwen_root: Path, check: bool = True) -> subprocess.CompletedProcess:
    compose_file = qwen_root / "docker-compose.yml"
    if shutil.which("docker-compose"):
        cmd = ["docker-compose", "-f", str(compose_file), *args]
    else:
        cmd = ["docker", "compose", "-f", str(compose_file), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(qwen_root), check=check)


def _normalize_intervals(segments: List[dict]) -> List[dict]:
    intervals = []
    for seg in segments:
        spk = seg.get("speaker", 0)
        try:
            spk_i = int(spk)
            label = f"SPEAKER_{spk_i:02d}"
        except (TypeError, ValueError):
            label = str(spk)
        intervals.append(
            {
                "speaker": label,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            }
        )
    return intervals


def _write_dual_markdown(
    path: Path,
    *,
    audio: Path,
    gigaam_txt: str,
    qwen_txt: str,
    gigaam_model: str,
    qwen_model: str,
) -> None:
    body = f"""# Dual transcript for LLM compile

Audio: `{audio}`

Primary (Sortformer + {gigaam_model}) is usually better on dialogue flow.
Secondary (Sortformer + {qwen_model}) is often better on rare names (e.g. Илюза).

Task for LLM: produce one clean Russian dialogue `[MM:SS] Спикер N: …`.
Prefer primary for structure/speakers; take secondary wording where it clearly fixes names/terms.
Do not invent content absent from both.

---

## A) GigaAM (primary)

```text
{gigaam_txt.rstrip()}
```

---

## B) Qwen3-ASR (secondary)

```text
{qwen_txt.rstrip()}
```
"""
    path.write_text(body, encoding="utf-8")


def run_qwen_dual(
    *,
    audio: Path,
    out_dir: Path,
    wav16: Path,
    diar_raw: Path,
    gigaam_transcript_txt: Path,
) -> Optional[Path]:
    """Run Qwen ASR on full wav, map words onto Sortformer segments, write dual files."""
    if not _truthy("QWEN_DUAL") and not _truthy("ENABLE_QWEN_DUAL"):
        return None

    qwen_root = qwen_project_root()
    if not (qwen_root / "docker-compose.yml").is_file():
        raise SystemExit(f"QWEN_PROJECT not found: {qwen_root}")

    mapper = _load_module("qwen_mapper", qwen_root / "app" / "mapper.py")
    diar_post = _load_module("qwen_diar_post", qwen_root / "app" / "diar_post.py")

    # Work under sister project's ./out (its compose mounts only that tree).
    dual_name = os.getenv("QWEN_DUAL_OUT", "").strip() or f"dual_{out_dir.name}"
    qwen_out = qwen_root / "out" / dual_name
    qwen_out.mkdir(parents=True, exist_ok=True)

    qwen_wav = qwen_out / "audio_16k.wav"
    qwen_diar = qwen_out / "diar.raw.json"
    qwen_asr = qwen_out / "asr.json"

    try:
        shutil.copy2(wav16, qwen_wav)
        shutil.copy2(diar_raw, qwen_diar)
    except PermissionError as exc:
        raise SystemExit(
            f"Cannot write into {qwen_out} (often root-owned Docker files). "
            f"Use a fresh QWEN_DUAL_OUT or: docker run --rm -v {qwen_out}:/o alpine "
            f"rm -rf /o/*\n{exc}"
        ) from exc

    wav_c = f"/out/{dual_name}/audio_16k.wav"
    asr_c = f"/out/{dual_name}/asr.json"

    skip_asr = _truthy("SKIP_QWEN_ASR") and qwen_asr.is_file()
    if skip_asr:
        print(f"[orch] skip Qwen ASR, reuse {qwen_asr}", flush=True)
    else:
        print("[orch] Qwen3-ASR (secondary pass)…", flush=True)
        _compose_qwen(
            "run",
            "--rm",
            "asr",
            "--audio",
            wav_c,
            "--out",
            asr_c,
            qwen_root=qwen_root,
        )

    asr = json.loads(qwen_asr.read_text(encoding="utf-8"))
    diar = json.loads(qwen_diar.read_text(encoding="utf-8"))
    raw_segments = diar.get("segments") or []

    max_spk = int(os.getenv("MAX_SPEAKERS", "2"))
    # Slightly tighter merge for word-mapping than GigaAM turn merge.
    merge_gap = float(os.getenv("QWEN_MAP_MERGE_GAP_SEC", "0.5"))
    min_seg = float(os.getenv("QWEN_MAP_MIN_SEGMENT_SEC", "0.25"))
    segments = diar_post.postprocess(
        raw_segments,
        max_speakers=max_spk,
        merge_gap=merge_gap,
        min_segment=min_seg,
    )
    intervals = _normalize_intervals(segments)

    words = asr.get("words") or []
    words = mapper.trim_words_to_speech(
        words,
        intervals,
        pad_sec=float(os.getenv("TRIM_PAD_SEC", "0.8")),
    )
    dialog = mapper.map_words_to_dialog(
        words,
        intervals,
        gap_tolerance=float(os.getenv("GAP_TOLERANCE_SEC", "0.5")),
        pause_split_sec=float(os.getenv("PAUSE_SPLIT_SEC", "0.65")),
    )
    dialog = mapper.apply_punctuation(asr.get("text") or "", dialog)
    dialog = mapper.drop_hallucinated_tail(dialog)
    qwen_txt = mapper.dialog_to_txt(dialog)

    # Copy artifacts into primary out dir
    transcript_qwen = out_dir / "transcript_qwen.txt"
    asr_qwen_json = out_dir / "asr_qwen.json"
    dialog_qwen_json = out_dir / "dialog_qwen.json"
    dual_md = out_dir / "dual_for_llm.md"

    transcript_qwen.write_text(qwen_txt, encoding="utf-8")
    shutil.copy2(qwen_asr, asr_qwen_json)
    dialog_qwen_json.write_text(
        json.dumps(
            {
                "audio": str(audio),
                "model": asr.get("model"),
                "aligner": asr.get("aligner"),
                "language": asr.get("language"),
                "full_text": asr.get("text"),
                "segments": dialog,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    gigaam_txt = (
        gigaam_transcript_txt.read_text(encoding="utf-8")
        if gigaam_transcript_txt.is_file()
        else ""
    )
    _write_dual_markdown(
        dual_md,
        audio=audio,
        gigaam_txt=gigaam_txt,
        qwen_txt=qwen_txt,
        gigaam_model=os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt"),
        qwen_model=str(asr.get("model") or "Qwen3-ASR"),
    )

    print(f"[orch] Qwen transcript → {transcript_qwen}", flush=True)
    print(f"[orch] dual for LLM → {dual_md}", flush=True)
    return dual_md
