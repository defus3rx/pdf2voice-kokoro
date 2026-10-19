# pdf2voice

Turn any PDF into narrated audio so you can listen while you read the same PDF.
It runs fully locally with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0),
uses your GPU when one is available, and works offline after the first model download.

The work is mostly in the text cleanup: running headers and footers, page numbers, the table of
contents, symbol tables, footnote markers and reference lists are removed, and the remaining
text is reflowed into paragraphs and rewritten so it sounds like a person reading, not a
machine reading the page layout.

## Setup

Requirements: a bash shell, `espeak-ng` and `ffmpeg`. Python 3.12 is installed for you by
[uv](https://docs.astral.sh/uv/). On Windows, use WSL.

```bash
./setup.sh
```

The script:

1. checks for `espeak-ng`, `ffmpeg` and `uv`, and asks before installing any that are missing
   (with Homebrew, apt, dnf or pacman, whichever your system has);
2. creates `.venv` (Python 3.12) and installs pdf2voice into it;
3. downloads the Kokoro weights and the default voice into the normal Hugging Face cache
   (`~/.cache/huggingface`), so later runs work without internet;
4. runs the unit tests;
5. offers to link `pdf2voice` into `~/.local/bin` so you can run it from any terminal.

If you'd rather use an alias, add this to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
alias pdf2voice='/path/to/pdf2voice/.venv/bin/pdf2voice'
```

## Usage

```bash
pdf2voice path/to/file.pdf [--out DIR] [--voice af_heart] [--speed 1.0]
          [--pages 10-55] [--split sections|single] [--format mp3|wav]
```

With no options it reads **all pages** with voice `af_heart` (American English) at speed 1.0,
and writes one MP3 per section into `<pdf-name>_audio/` next to the PDF.

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | `<pdf-name>_audio/` next to the PDF | output folder |
| `--voice NAME` | `af_heart` | Kokoro voice (`--list-voices` shows them all) |
| `--lang CODE` | from the voice name (`a`) | Kokoro language code |
| `--speed X` | `1.0` | speaking speed, 0.5 to 2.0 |
| `--pages SPEC` | all pages | e.g. `10-55`, `1-5,8,20-` |
| `--split sections\|single` | `sections` | one file per section, or one file with chapter markers |
| `--format mp3\|wav` | `mp3` | MP3 is 64 kbps mono, which is plenty for speech |
| `--bitrate` | `64k` | MP3 bitrate |
| `--keep-references` | off | also read reference and bibliography lists |
| `--text-only` | off | stop after writing `cleaned.txt` |
| `--device auto\|cuda\|mps\|cpu` | `auto` | GPU when available, otherwise CPU |
| `--force` | off | ignore the cache and regenerate everything |
| `--list-voices` | | list voices (`*` marks ones already downloaded) |

Examples:

```bash
pdf2voice report.pdf --text-only            # check what will be spoken first
pdf2voice report.pdf --pages 10-55 --speed 1.1
pdf2voice report.pdf --split single         # one MP3 with chapters
pdf2voice report.pdf --voice bm_george      # British male voice (lang b picked automatically)
```

### Output

```
report_audio/
  01_Front_matter.mp3
  02_Introduction.mp3
  03_LLM01-2026_Prompt_Injection.mp3
  ...
  playlist.m3u      # all files in order, with titles and durations
  index.md          # file, section title, starting PDF page, duration
  cleaned.txt       # exactly the text that is spoken
  .cache/           # finished sections (FLAC), used for resuming
```

With `--split single` you get one `<pdf-title>.mp3` with ID3 chapter markers (VLC and most
podcast and audiobook players show them), and `index.md` lists each chapter's start
time and PDF page. WAV files can't carry chapters, so with `--format wav` you only get the
times in `index.md`.

### Resuming

Each finished section is cached under `.cache/`, keyed by a hash of its spoken text, the voice,
the language and the speed. Re-running the same command skips every section whose text hasn't
changed, so an interrupted run (Ctrl-C) continues where it stopped, and editing the cleanup
rules only regenerates the sections they affect. Audio files from an earlier run that the
current run no longer produces (for example after a section was renamed) are removed.

## How the text is cleaned

| Problem in the PDF | What pdf2voice does |
|---|---|
| Running headers/footers ("genai.owasp.org", "Page 12") | Lines in the top or bottom 10% of the page that repeat on at least 40% of pages are dropped, with digits normalized so "Page 3" and "Page 4" match. Standalone page numbers are dropped too. |
| Table of contents | Pages titled "Contents" or mostly made of "title … page" lines are skipped, as are outline entries named "Table of Contents". |
| Tables and matrices | Tables found by PyMuPDF's table detector, and rows that are mostly symbols (●, ○, —, ✓), are replaced with one spoken "Table skipped." |
| Hyphenated line breaks | `exam-` + `ple` becomes `example`, but `prompt-` + `injection` stays `prompt-injection`, decided using the system dictionary plus the document's own vocabulary. |
| Wrapped lines | Lines are reflowed into paragraphs using line spacing, font size and bullets, including paragraphs that continue across page breaks. Each paragraph break becomes a 300 ms pause. |
| URLs, emails, hashes | `https://creativecommons.org/licenses/...` is read as "creativecommons dot org", an email as "name at domain dot com"; hashes and UUIDs are dropped. |
| Abbreviations and IDs | "e.g." becomes "for example", "i.e." becomes "that is", "et al." becomes "and colleagues", `LLM01:2026` becomes "L L M zero one, twenty twenty-six", `2025-12-09` becomes "December 9, twenty twenty-five", `#3` becomes "number 3". |
| Bullets, footnote markers, figure placeholders | Removed. Figure captions are kept. |
| Reference lists | Sections or subsections titled References, Bibliography, Works Cited and so on are skipped unless you pass `--keep-references`. |

**Sections**: if the PDF has an outline (bookmarks), its entries are matched to headings on the
page and the largest heading tier becomes the section boundary. This handles flat outlines
where "Description" sits next to "LLM01" at the same level. Without an outline, the largest
recurring font size is used.

## Audio pipeline

* Text is fed to Kokoro in sentence-aware chunks of at most about 350 characters. A chunk never
  ends mid-sentence; an overlong sentence is split at a `;`, `:` or `,`.
* The audio is 24 kHz mono, with about 80 ms between chunks, 300 ms between paragraphs,
  600 ms around headings and 800 ms between sections.
* A GPU is used when available (CUDA or MPS), falling back to the CPU automatically if
  anything fails on the GPU. On a laptop GPU, a 122-page document ran at about 7x realtime:
  2 h 28 min of audio took 22 minutes.
* Once the model and voice are in the Hugging Face cache, `HF_HUB_OFFLINE=1` is set
  automatically, so no network calls are made.

## Development

```bash
.venv/bin/python -m pytest -q
```

Layout:

```
pdf2voice/
  textclean.py   pure string functions (unit tested): headers, TOC, tables, hyphens, speakable text, chunking
  extract.py     PyMuPDF layout → paragraphs → sections
  tts.py         Kokoro wrapper, device selection, voice list, offline cache detection
  audio.py       FLAC cache, ffmpeg MP3 encoding, chapter markers
  cli.py         command line, caching/resume, playlist and index
tests/test_textclean.py
```

## Limitations

* Scanned PDFs without a text layer aren't supported (no OCR).
* Multi-column layouts follow the PDF's content order, which is usually right but not always.
* The text normalization (abbreviations, IDs, dates) is English-only. Other languages get the
  layout cleanup but not the English rewrites.

## License

MIT, see [LICENSE](LICENSE). Kokoro-82M is Apache-2.0 and is downloaded separately; PyMuPDF is AGPL-3.0.
