"""pdf2voice command line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from . import __version__  # noqa: E402
from .extract import Document, Section, load_document  # noqa: E402
from .tts import LANGUAGES, SAMPLE_RATE, VOICES, cached_voices, default_lang_for  # noqa: E402

CACHE_VERSION = 1
PARAGRAPH_GAP_MS = 300
HEADING_GAP_MS = 600
CHUNK_GAP_MS = 80
SECTION_GAP_MS = 800


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf2voice",
        description="Turn a PDF into narrated audio with Kokoro-82M, locally and offline.",
    )
    p.add_argument("pdf", nargs="?", type=Path, help="PDF file to narrate")
    p.add_argument("--out", type=Path, help="output folder (default: <pdf-name>_audio/ next to the PDF)")
    p.add_argument("--voice", default="af_heart", help="Kokoro voice, see --list-voices (default: af_heart)")
    p.add_argument("--lang", default=None, help="Kokoro lang_code (default: from the voice name, 'a' for af_heart)")
    p.add_argument("--speed", type=float, default=1.0, help="speaking speed, 0.5-2.0 (default: 1.0)")
    p.add_argument("--pages", default=None, help="pages to read, e.g. 10-55 or 1-5,8,20- (default: all pages)")
    p.add_argument("--split", choices=["sections", "single"], default="sections",
                   help="one file per section (default), or one file with chapter markers")
    p.add_argument("--format", choices=["mp3", "wav"], default="mp3", help="output audio format (default: mp3)")
    p.add_argument("--bitrate", default="64k", help="mp3 bitrate; 64k-96k is plenty for speech (default: 64k)")
    p.add_argument("--keep-references", action="store_true", help="read reference/bibliography lists too")
    p.add_argument("--text-only", action="store_true", help="only write cleaned.txt, no audio")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                   help="torch device (default: auto = GPU if available, else CPU)")
    p.add_argument("--force", action="store_true", help="ignore the cache and regenerate everything")
    p.add_argument("--list-voices", action="store_true", help="list available voices and exit")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def list_voices() -> None:
    have = cached_voices()
    print("Kokoro-82M voices (* = already downloaded, usable offline)\n")
    for lang, voices in VOICES.items():
        print(f"  --lang {lang}  {LANGUAGES[lang]}")
        for v in voices:
            print(f"      {'*' if v in have else ' '} {v}")
    print("\nThe first letter of a voice is its language; the second is f(emale)/m(ale).")


def install_hint(pkg: str) -> str:
    for tool, cmd in (("brew", "brew install"), ("apt-get", "sudo apt install"), ("dnf", "sudo dnf install"),
                      ("pacman", "sudo pacman -S"), ("winget", "winget install")):
        if shutil.which(tool):
            return f"Install it with:  {cmd} {pkg}"
    return f"Install {pkg} with your system's package manager."


def safe_name(title: str, maxlen: int = 80) -> str:
    s = title.replace(":", "-").replace("&", "and").replace("/", "-")
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.ASCII)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:maxlen].rstrip("._-") or "section"


def fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class Item:
    index: int
    section: Section
    key: str
    cache: Path
    output: Path | None  # per-section output file (split mode)

    @property
    def chars(self) -> int:
        return sum(len(t) for _, t in self.section.spoken)


def section_key(sec: Section, voice: str, lang: str, speed: float) -> str:
    payload = json.dumps({"v": CACHE_VERSION, "voice": voice, "lang": lang, "speed": speed,
                          "gaps": [PARAGRAPH_GAP_MS, HEADING_GAP_MS, CHUNK_GAP_MS, SECTION_GAP_MS],
                          "text": sec.spoken}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def write_cleaned(doc: Document, path: Path) -> None:
    parts = []
    for sec in doc.sections:
        parts.append(f"=== {sec.title} ===")
        parts.extend(t for _, t in sec.spoken)
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def synthesize_section(narrator, item: Item, total_bar, tqdm):
    import numpy as np

    from .tts import silence

    pieces = []
    plan = []  # (kind, chunk list)
    for kind, text in item.section.spoken:
        plan.append((kind, narrator.chunks(text)))
    n_chunks = sum(len(c) for _, c in plan)
    title = item.section.title if len(item.section.title) <= 40 else item.section.title[:39] + "…"
    with tqdm(total=n_chunks, desc=f"  {item.index:02d} {title}", unit="chunk", leave=False,
              position=1, dynamic_ncols=True) as bar:
        for p_idx, (kind, chunks) in enumerate(plan):
            if p_idx:
                gap = HEADING_GAP_MS if kind == "heading" or plan[p_idx - 1][0] == "heading" else PARAGRAPH_GAP_MS
                pieces.append(silence(gap))
            for c_idx, chunk in enumerate(chunks):
                if c_idx:
                    pieces.append(silence(CHUNK_GAP_MS))
                pieces.append(narrator.synthesize(chunk))
                bar.update(1)
                total_bar.update(len(chunk))
    pieces.append(silence(SECTION_GAP_MS))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_voices:
        list_voices()
        return 0
    if args.pdf is None:
        build_parser().print_usage()
        print("pdf2voice: error: a PDF path is required (or use --list-voices)")
        return 2

    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file():
        print(f"pdf2voice: error: no such file: {pdf}")
        return 2
    lang = args.lang or default_lang_for(args.voice)
    if lang not in LANGUAGES:
        print(f"pdf2voice: error: unknown --lang {lang!r}; choose from {', '.join(LANGUAGES)}")
        return 2
    if args.voice[:1] in LANGUAGES and args.voice[0] != lang:
        print(f"Warning: voice {args.voice} is a {LANGUAGES[args.voice[0]]} voice but --lang is {lang} "
              f"({LANGUAGES[lang]}); pronunciation may be odd.")
    if not 0.5 <= args.speed <= 2.0:
        print("pdf2voice: error: --speed must be between 0.5 and 2.0")
        return 2

    out = (args.out.expanduser() if args.out else pdf.parent / f"{pdf.stem}_audio").resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.text_only:
        if args.format == "mp3" and not shutil.which("ffmpeg"):
            print(f"pdf2voice: error: ffmpeg is needed for mp3 output. {install_hint('ffmpeg')}\n"
                  "(or use --format wav)")
            return 1
        if not shutil.which("espeak-ng"):
            print("Warning: espeak-ng not found; Kokoro uses it for words outside its dictionary. "
                  + install_hint("espeak-ng"))

    t0 = time.time()
    print(f"Reading {pdf.name} …")
    try:
        doc = load_document(pdf, args.pages, keep_references=args.keep_references, lang=lang)
    except ValueError as exc:
        print(f"pdf2voice: error: {exc}")
        return 2
    cleaned = out / "cleaned.txt"
    write_cleaned(doc, cleaned)
    words = sum(s.word_count for s in doc.sections)
    pages_desc = "all" if len(doc.pages) == doc.page_count else args.pages
    print(f"  {doc.page_count} pages (reading: {pages_desc}), {len(doc.sections)} sections, {words:,} words "
          f"[{time.time() - t0:.1f}s]")
    print(f"  Headings from: {doc.heading_source}")
    for s in doc.skipped:
        print(f"  Skipped: {s}")
    print(f"  Cleaned text: {cleaned}")
    if not doc.sections:
        print("No readable text found (is this a scanned PDF without a text layer?).")
        return 1
    if args.text_only:
        for i, s in enumerate(doc.sections, 1):
            print(f"  {i:02d}  p{s.page + 1:<4} {s.word_count:6,}w  {s.title}")
        return 0

    return generate(doc, out, args, lang)


def generate(doc: Document, out: Path, args, lang: str) -> int:
    from tqdm import tqdm

    from . import audio
    from .tts import Narrator

    cache_dir = out / ".cache"
    cache_dir.mkdir(exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    all_outputs: dict[str, dict[str, str]] = manifest.get("outputs", {})
    if not all(isinstance(v, dict) for v in all_outputs.values()):
        all_outputs = {}
    previous_outputs = all_outputs.get(args.split, {})

    ext = args.format
    width = 2 if len(doc.sections) < 100 else 3
    items: list[Item] = []
    for i, sec in enumerate(doc.sections, 1):
        key = section_key(sec, args.voice, lang, args.speed)
        output = out / f"{i:0{width}d}_{safe_name(sec.title)}.{ext}" if args.split == "sections" else None
        items.append(Item(i, sec, key, cache_dir / f"{key}.flac", output))
    single_out = out / f"{safe_name(doc.title) or 'audiobook'}.{ext}" if args.split == "single" else None
    encode_sig = f"{args.format}:{args.bitrate}:{len(items)}"

    def output_done(item: Item) -> bool:
        return (not args.force and item.output is not None and item.output.exists()
                and previous_outputs.get(item.output.name) == f"{item.key}:{encode_sig}")

    todo = [it for it in items if args.force or not it.cache.exists()]
    todo_ids = {it.index for it in todo}
    t_start = time.time()
    narrator = None
    synthesized = 0
    album = doc.title
    artist = f"pdf2voice · Kokoro {args.voice}"
    new_outputs: dict[str, str] = {}

    def save_manifest():
        all_outputs[args.split] = new_outputs
        manifest_path.write_text(json.dumps({"version": CACHE_VERSION, "outputs": all_outputs}, indent=1))

    if todo:
        print("Loading Kokoro-82M …")
        narrator = Narrator(args.voice, lang, args.speed, args.device)
        dev_name = {"cuda": "GPU (CUDA)", "mps": "GPU (MPS)", "cpu": "CPU"}[narrator.device]
        print(f"  Device: {dev_name}   Voice: {args.voice}   Lang: {lang} ({LANGUAGES[lang]})   "
              f"Speed: {args.speed}" + ("   [offline, using cached model]" if narrator.offline else ""))
    else:
        print("All sections are cached; nothing to synthesize.")

    cached_count = len(items) - len(todo)
    if cached_count and todo:
        print(f"  Reusing {cached_count} cached section(s); synthesizing {len(todo)}.")

    total_chars = sum(it.chars for it in todo)
    try:
        with tqdm(total=total_chars, desc="Total", unit="char", unit_scale=True, position=0,
                  dynamic_ncols=True, disable=not todo,
                  bar_format="{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]") as total_bar:
            for it in items:
                if it.index in todo_ids:
                    pcm = synthesize_section(narrator, it, total_bar, tqdm)
                    audio.write_flac(it.cache, pcm)
                    synthesized += 1
                if it.output is not None:
                    if not output_done(it):
                        tags = {"title": it.section.title, "album": album, "artist": artist,
                                "track": f"{it.index}/{len(items)}", "genre": "Speech"}
                        audio.encode(it.cache, it.output, args.format, args.bitrate, tags)
                    new_outputs[it.output.name] = f"{it.key}:{encode_sig}"
                    save_manifest()
    except KeyboardInterrupt:
        save_manifest()
        print("\nInterrupted. Finished sections are cached; run the same command again to resume.")
        return 130

    chapters = None
    if single_out is not None:
        print(f"Joining {len(items)} sections into {single_out.name} …")
        chapters = audio.concat_with_chapters(
            [(it.section.title, it.cache) for it in items], single_out, args.format, args.bitrate,
            {"title": album, "album": album, "artist": artist, "genre": "Speech"}, gap_ms=0)
        new_outputs[single_out.name] = f"single:{encode_sig}"

    # Remove audio files from earlier runs that this run no longer produces (e.g. renamed sections).
    for name in previous_outputs:
        if name not in new_outputs and (out / name).exists():
            (out / name).unlink()
    save_manifest()

    durations = {it.index: audio.duration(it.cache) for it in items}
    write_playlist(out, items, durations, single_out, chapters)
    write_index(out, doc, items, durations, single_out, chapters, args, lang)

    elapsed = time.time() - t_start
    total_audio = chapters[-1].end if chapters else sum(durations.values())
    print("\nDone.")
    print(f"  Output:        {out}")
    print(f"  Files:         {single_out.name if single_out else f'{len(items)} section files'}, "
          "playlist.m3u, index.md, cleaned.txt")
    print(f"  Audio length:  {fmt_dur(total_audio)}")
    print(f"  Time taken:    {fmt_dur(elapsed)} ({synthesized} section(s) synthesized, "
          f"{len(items) - synthesized} from cache)")
    if narrator is not None:
        speedup = sum(durations[it.index] for it in todo) / max(elapsed, 1e-6)
        print(f"  Device:        {narrator.device}  (~{speedup:.1f}x faster than realtime)")
    return 0


def write_playlist(out: Path, items: list[Item], durations: dict[int, float], single_out: Path | None,
                   chapters) -> None:
    lines = ["#EXTM3U"]
    if single_out is not None:
        lines += [f"#EXTINF:{int(sum(durations.values()))},{single_out.stem}", single_out.name]
    else:
        for it in items:
            lines += [f"#EXTINF:{int(durations[it.index])},{it.section.title}", it.output.name]
    (out / "playlist.m3u").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(out: Path, doc: Document, items: list[Item], durations: dict[int, float],
                single_out: Path | None, chapters, args, lang: str) -> None:
    pages = "all pages" if len(doc.pages) == doc.page_count else f"pages {args.pages}"
    lines = [f"# {doc.title}: audio index", "",
             f"Source: `{doc.path.name}` ({pages}). Voice `{args.voice}` (lang `{lang}`), speed {args.speed}.",
             f"Generated by pdf2voice {__version__} with Kokoro-82M.", ""]
    if single_out is not None:
        lines += [f"File: [{single_out.name}]({single_out.name})"
                  + (" (chapter markers embedded)" if args.format == "mp3" else ""), "",
                  "| # | Chapter | Starts at | PDF page |", "|---:|---|---:|---:|"]
        for it, ch in zip(items, chapters):
            lines.append(f"| {it.index} | {it.section.title} | {fmt_dur(ch.start)} | {it.section.page + 1} |")
    else:
        lines += ["| # | File | Section | PDF page | Duration |", "|---:|---|---|---:|---:|"]
        for it in items:
            lines.append(f"| {it.index} | [{it.output.name}]({it.output.name}) | {it.section.title} | "
                         f"{it.section.page + 1} | {fmt_dur(durations[it.index])} |")
    lines += ["", f"Total: {fmt_dur(sum(durations.values()))}"]
    if doc.skipped:
        lines += ["", "Not narrated:"] + [f"- {s}" for s in doc.skipped]
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
