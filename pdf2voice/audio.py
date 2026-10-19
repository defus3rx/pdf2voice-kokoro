"""Audio file helpers: FLAC section cache, ffmpeg encoding, chapter markers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from .tts import SAMPLE_RATE, silence


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def write_flac(path: Path, audio: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.flac")
    sf.write(tmp, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    tmp.replace(path)


def duration(path: Path) -> float:
    return sf.info(str(path)).duration


def _escape_meta(s: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        s = s.replace(ch, "\\" + ch)
    return s


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_path() or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def encode(src: Path, dst: Path, fmt: str, bitrate: str, tags: dict[str, str],
           chapters_file: Path | None = None) -> None:
    """Encode a cached FLAC/WAV into the final mp3 or wav, atomically."""
    tmp = dst.with_name(dst.stem + ".partial" + dst.suffix)
    if fmt == "wav":
        data, sr = sf.read(str(src), dtype="int16")
        sf.write(tmp, data, sr, subtype="PCM_16")
    else:
        args = ["-i", str(src)]
        if chapters_file:
            args += ["-i", str(chapters_file), "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
        args += ["-ac", "1", "-c:a", "libmp3lame", "-b:a", bitrate, "-id3v2_version", "3"]
        for k, v in tags.items():
            args += ["-metadata", f"{k}={v}"]
        args.append(str(tmp))
        _run_ffmpeg(args)
    tmp.replace(dst)


@dataclass
class Chapter:
    title: str
    start: float
    end: float


def concat_with_chapters(parts: Sequence[tuple[str, Path]], dst: Path, fmt: str, bitrate: str,
                         tags: dict[str, str], gap_ms: int = 800) -> list[Chapter]:
    """Join section FLACs into one file (with ID3 chapter markers for mp3)."""
    chapters: list[Chapter] = []
    gap = silence(gap_ms)
    with tempfile.TemporaryDirectory(dir=dst.parent) as td:
        joined = Path(td) / "joined.wav"
        pos = 0
        with sf.SoundFile(str(joined), "w", SAMPLE_RATE, 1, "PCM_16") as out:
            for i, (title, path) in enumerate(parts):
                if i:
                    out.write(gap)
                    pos += len(gap)
                data, _ = sf.read(str(path), dtype="float32")
                chapters.append(Chapter(title, pos / SAMPLE_RATE, (pos + len(data)) / SAMPLE_RATE))
                out.write(data)
                pos += len(data)
        meta = Path(td) / "chapters.txt"
        lines = [";FFMETADATA1", f"title={_escape_meta(tags.get('title', ''))}"]
        for c in chapters:
            lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(c.start * 1000)}", f"END={int(c.end * 1000)}",
                      f"title={_escape_meta(c.title)}"]
        meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
        encode(joined, dst, fmt, bitrate, tags, chapters_file=meta if fmt == "mp3" else None)
    return chapters
