"""Audio helpers: ffmpeg resample to 16 kHz mono, optional enhance, cut segments."""

from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path
from typing import Optional, Tuple


def enhance_mode_from_env() -> str:
    """Return: off | highpass | full (highpass+loudnorm)."""
    raw = os.getenv("AUDIO_ENHANCE", "highpass").strip().lower()
    if raw in {"0", "false", "no", "off", ""}:
        return "off"
    if raw in {"highpass", "hp"}:
        return "highpass"
    if raw in {"1", "true", "yes", "full", "loudnorm"}:
        return "full"
    return "off"


def _ffmpeg_to_wav16(
    src: Path,
    dst: Path,
    *,
    highpass_hz: float = 0.0,
    loudnorm: bool = False,
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    filters = ["aresample=16000", "aformat=sample_fmts=s16:channel_layouts=mono"]
    af_parts = []
    if highpass_hz and highpass_hz > 0:
        af_parts.append(f"highpass=f={highpass_hz:g}")
    if loudnorm:
        af_parts.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    af_parts.extend(filters)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        ",".join(af_parts),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")
    return dst


def ensure_wav_16k_mono(
    src: Path,
    dst: Path,
    *,
    mode: Optional[str] = None,
    highpass_hz: Optional[float] = None,
) -> Path:
    """Resample to 16 kHz mono PCM.

    mode: off | highpass | full (from AUDIO_ENHANCE if None).
    """
    if mode is None:
        mode = enhance_mode_from_env()
    if highpass_hz is None:
        highpass_hz = float(os.getenv("AUDIO_HIGHPASS_HZ", "80"))

    if mode == "highpass":
        return _ffmpeg_to_wav16(src, dst, highpass_hz=highpass_hz, loudnorm=False)
    if mode == "full":
        return _ffmpeg_to_wav16(src, dst, highpass_hz=highpass_hz, loudnorm=True)

    # off: fast path if already 16k mono pcm_s16le
    try:
        with wave.open(str(src), "rb") as handle:
            if (
                handle.getframerate() == 16000
                and handle.getnchannels() == 1
                and handle.getsampwidth() == 2
            ):
                if src.resolve() != dst.resolve():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())
                return dst
    except wave.Error:
        pass

    return _ffmpeg_to_wav16(src, dst, highpass_hz=0.0, loudnorm=False)


def cut_segment(src_wav: Path, dst_wav: Path, start: float, end: float) -> Path:
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src_wav),
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "pcm_s16le",
        str(dst_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst_wav


def format_ts(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
