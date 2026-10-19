"""Pure string functions that turn PDF text into something that sounds right read aloud.

Everything here works on plain strings so it can be unit-tested without a PDF.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

TABLE_CUE = "Table skipped."

# Glyphs that fill the cells of comparison / mapping matrices.
TABLE_SYMBOLS = set("●○◐◑◒◓◔◕◉◎•◦✓✔✗✘✕✖×—–-■□▪▫◆◇★☆✅❌➖➕+?")
BULLET_GLYPHS = "•●◦▪▫‣⁃∙·○■□➢➤►▶✓✔-–—*"
BULLET_RE = re.compile(rf"^\s*[{re.escape(BULLET_GLYPHS)}](?:\s+|$)")

_ROMAN = r"(?=[ivxlcdm])m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})"
PAGE_NUMBER_RE = re.compile(
    rf"^\s*[-–—]?\s*(?:(?:page|p\.|pg\.?)\s*)?(?:\d{{1,4}}|{_ROMAN})(?:\s*(?:of|/)\s*\d{{1,4}})?\s*[-–—]?\s*$",
    re.IGNORECASE,
)
TOC_LEADER_RE = re.compile(r"^.{2,}?(?:\s*[.·•…_]\s*){3,}\s*\d{1,4}\s*$")
TOC_GAP_RE = re.compile(r"^\D.{1,}?\S(?:\s{2,}|\t+)\d{1,4}\s*$")
TOC_LOOSE_RE = re.compile(r"^[^.!?]{2,100}?[A-Za-z)\]:]\s+\d{1,4}\s*$")
TOC_TITLE_RE = re.compile(r"^\s*(?:table\s+of\s+)?contents?\s*$", re.IGNORECASE)
REFERENCES_TITLE_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:references?|reference\s+links|bibliography|works\s+cited|"
    r"citations|sources|further\s+reading|related\s+(?:resources|reading))\s*:?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Line classification
# ---------------------------------------------------------------------------

def is_page_number(line: str) -> bool:
    """A line that is nothing but a page number ("12", "Page 12", "12 of 80", "xii")."""
    return bool(PAGE_NUMBER_RE.match(line.strip())) if line.strip() else False


def is_toc_line(line: str, loose: bool = False) -> bool:
    """A table-of-contents entry: dotted leaders, or a title followed by a page number.

    ``loose`` also accepts "Title 12" with a single space, which is only safe when most
    lines on the page look like that (see ``looks_like_toc_page``).
    """
    s = line.strip()
    if not s or is_page_number(s):
        return False
    if TOC_LEADER_RE.match(s) or TOC_GAP_RE.match(line.rstrip()):
        return True
    return loose and bool(TOC_LOOSE_RE.match(s))


def looks_like_toc_page(lines: Sequence[str], min_ratio: float = 0.5) -> bool:
    lines = [l for l in lines if l.strip()]
    if any(TOC_TITLE_RE.match(l) for l in lines[:5]):
        return True
    if len(lines) < 4:
        return False
    hits = sum(is_toc_line(l, loose=True) for l in lines)
    return hits / len(lines) >= min_ratio


def is_table_row(line: str) -> bool:
    """A row of a symbol matrix, e.g. ``LLM01  ●  ○  ●  —  ✓``."""
    tokens = line.split()
    if not tokens:
        return False
    sym = sum(1 for t in tokens if all(c in TABLE_SYMBOLS for c in t))
    return sym >= 3 or (sym >= 2 and sym / len(tokens) >= 0.5)


def _is_short_label(line: str) -> bool:
    s = line.strip()
    return bool(s) and len(s.split()) <= 4 and s[-1] not in ".!?:;"


def skip_tables(lines: Sequence[str], cue: str = TABLE_CUE) -> list[str]:
    """Replace each run of symbol-table rows with a single spoken cue.

    Short label lines sandwiched between table rows (wrapped row labels such as
    "Unbounded" / "Consumption") and a column-header line directly above the table
    are treated as part of it.
    """
    n = len(lines)
    mark = [is_table_row(l) for l in lines]
    # Fill short gaps between table rows.
    idx = [i for i, m in enumerate(mark) if m]
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= 5 and all(_is_short_label(lines[k]) or not lines[k].strip() for k in range(a + 1, b)):
            for k in range(a + 1, b):
                mark[k] = True
    # A column-header row directly above the first table row of each run.
    for i in range(1, n):
        if mark[i] and not mark[i - 1] and _looks_like_header_row(lines[i - 1]):
            mark[i - 1] = True
    out: list[str] = []
    for i, line in enumerate(lines):
        if mark[i]:
            if not out or out[-1] != cue:
                out.append(cue)
        else:
            out.append(line)
    return out


def _looks_like_header_row(line: str) -> bool:
    words = line.split()
    return len(words) >= 5 and line.strip()[-1:] not in ".!?:;," and sum(len(w) for w in words) / len(words) <= 7


def find_repeated_lines(pages: Sequence[Sequence[str]], min_ratio: float = 0.5, edge: int = 3) -> set[str]:
    """Normalized forms of header/footer lines repeated on most pages.

    Only the first and last ``edge`` lines of each page are candidates. Digits are
    normalized so that "Page 3" and "Page 4" count as the same footer.
    """
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        lines = [l for l in lines if l.strip()]
        cands = {normalize_for_repeat(l) for l in lines[:edge] + lines[-edge:]}
        counts.update(c for c in cands if c)
    threshold = max(2, int(len(pages) * min_ratio + 0.5))
    return {k for k, c in counts.items() if c >= threshold}


def normalize_for_repeat(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def remove_headers_footers(pages: Sequence[Sequence[str]], min_ratio: float = 0.5, edge: int = 3) -> list[list[str]]:
    """Drop repeated headers/footers and standalone page numbers from each page."""
    repeated = find_repeated_lines(pages, min_ratio, edge)
    out = []
    for lines in pages:
        nonblank = [i for i, l in enumerate(lines) if l.strip()]
        edges = set(nonblank[:edge] + nonblank[-edge:])
        kept = []
        for i, l in enumerate(lines):
            if i in edges and (normalize_for_repeat(l) in repeated or is_page_number(l)):
                continue
            kept.append(l)
        out.append(kept)
    return out


# ---------------------------------------------------------------------------
# Hyphenation and reflow
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def system_words() -> frozenset[str]:
    for p in ("/usr/share/dict/words", "/usr/dict/words"):
        path = Path(p)
        if path.exists():
            return frozenset(w.strip().lower() for w in path.read_text(errors="ignore").splitlines() if w.strip())
    return frozenset()


_SUFFIXES = ("s", "es", "ed", "d", "ing", "ly", "er", "ers", "al", "ally", "ity", "ation", "ations", "ness", "ment", "ments")


def in_vocab(word: str, vocab: Iterable[str] | None = None) -> bool:
    w = word.lower()
    if len(w) < 2:
        return False
    sets = [system_words()] + ([vocab] if vocab is not None else [])
    for s in sets:
        if w in s:
            return True
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3 and (w[: -len(suf)] in s or w[: -len(suf)] + "e" in s):
                return True
    return False


_LEFT_WORD = re.compile(r"([A-Za-z]+)-$")
_RIGHT_WORD = re.compile(r"^([A-Za-z]+)")


def join_hyphen(left: str, right: str, vocab: Iterable[str] | None = None) -> str:
    """Join two line fragments where ``left`` ends with a hyphen.

    Decides between a soft line-break hyphen ("exam-" + "ple" -> "example") and a
    real compound ("prompt-" + "injection" -> "prompt-injection").
    """
    m1, m2 = _LEFT_WORD.search(left), _RIGHT_WORD.match(right)
    if not (m1 and m2 and m2.group(1)[0].islower()):
        return left + right
    w1, w2 = m1.group(1), m2.group(1)
    if in_vocab(w1 + w2, vocab):
        return left[:-1] + right
    if in_vocab(w1, vocab) and in_vocab(w2, vocab):
        return left + right
    return left[:-1] + right


_URL_START = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)


def join_lines(left: str, right: str, vocab: Iterable[str] | None = None) -> str:
    """Append a wrapped line to a paragraph."""
    left, right = left.rstrip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    last = left.split()[-1]
    if _URL_START.match(last) and last[-1] in "/_=&?#-":
        return left + right
    if left.endswith("-") and len(left) >= 2 and left[-2].isalpha():
        return join_hyphen(left, right, vocab)
    if left.endswith("-") and len(left) >= 2 and left[-2].isdigit() and right[0].isdigit():
        return left + right
    return left + " " + right


def join_hyphenated(text: str, vocab: Iterable[str] | None = None) -> str:
    """Join words broken across lines with a hyphen, e.g. ``exam-\\nple`` -> ``example``."""
    lines = text.split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if out[-1].rstrip().endswith("-") and line.strip():
            out[-1] = join_hyphen(out[-1].rstrip(), line.lstrip(), vocab)
        else:
            out.append(line)
    return "\n".join(out)


def reflow(lines: Sequence[str], vocab: Iterable[str] | None = None) -> list[str]:
    """Reflow wrapped lines into paragraphs.

    Blank lines and bullet items start new paragraphs; everything else is joined.
    """
    paras: list[str] = []
    cur = ""
    for line in lines:
        if not line.strip():
            if cur:
                paras.append(cur)
            cur = ""
            continue
        if line.strip() == TABLE_CUE or BULLET_RE.match(line):
            if cur:
                paras.append(cur)
            cur = line.strip()
            if line.strip() == TABLE_CUE:
                paras.append(cur)
                cur = ""
            continue
        cur = join_lines(cur, line, vocab)
    if cur:
        paras.append(cur)
    return paras


def vocab_from_text(texts: Iterable[str], min_count: int = 2) -> set[str]:
    c: Counter[str] = Counter()
    for t in texts:
        c.update(w.lower() for w in re.findall(r"[A-Za-z]{3,}", t))
    return {w for w, n in c.items() if n >= min_count}


# ---------------------------------------------------------------------------
# Making text speakable
# ---------------------------------------------------------------------------

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_MONTHS = ("January February March April May June July August September October November December").split()


def two_digit_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")


def year_words(y: int) -> str:
    """2026 -> 'twenty twenty-six', 2005 -> 'two thousand five', 1900 -> 'nineteen hundred'."""
    if 2000 <= y <= 2009:
        return "two thousand" + (" " + _ONES[y % 10] if y % 10 else "")
    if not 1000 <= y <= 9999:
        return str(y)
    hi, lo = divmod(y, 100)
    if lo == 0:
        return two_digit_words(hi) + " hundred"
    return two_digit_words(hi) + (" oh " + _ONES[lo] if lo < 10 else " " + two_digit_words(lo))


def digits_words(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def spell_letters(s: str) -> str:
    return " ".join(s)


def speak_domain(host: str) -> str:
    host = host.lower().removeprefix("www.")
    return " dot ".join(p for p in host.split(".") if p)


_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()\[\]\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b")
_TLDS = "com|org|net|io|ai|dev|gov|edu|co|uk|us|eu|de|fr|info|app|cloud|tech|security"
_DOMAIN_RE = re.compile(rf"\b(?<![@/.\w-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:{_TLDS}))\b(?![\w-])", re.IGNORECASE)
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HEX_RE = re.compile(r"\b(?:0x)?(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{16,}\b")
_LONG_TOKEN_RE = re.compile(r"(?<![\w/])(?=[A-Za-z0-9+/_=-]*\d)(?=[A-Za-z0-9+/_=-]*[A-Za-z])[A-Za-z0-9+/_-]{32,}={0,2}(?![\w/])")
_CODE_ID_RE = re.compile(r"\b([A-Z]{2,6})(\d{2,3})(?::(\d{4}))?\b")
_CVE_RE = re.compile(r"\bCVE-(\d{4})-(\d{4,7})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_FOOTNOTE_BRACKET_RE = re.compile(r"(?<=\S)\s?\[(?:\d{1,3}(?:\s*[-–,]\s*\d{1,3})*)\]")
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾†‡"
_FIGURE_PLACEHOLDER_RE = re.compile(r"^\s*[\[<(]?\s*(?:figure|fig\.?|image|img|diagram|chart|graphic|picture)(?:\s*\d+(?:\.\d+)*)?\s*[\]>)]?\s*$", re.IGNORECASE)
_INVISIBLE_RE = re.compile("[­​‌‍⁠﻿]")

_ABBREVIATIONS = [
    (re.compile(r"\be\.\s?g\.(?:,)?", re.IGNORECASE), "for example,"),
    (re.compile(r"\bi\.\s?e\.(?:,)?", re.IGNORECASE), "that is,"),
    (re.compile(r"\betc\.(?=\s+[a-z])"), "et cetera"),
    (re.compile(r"\betc\."), "et cetera."),
    (re.compile(r"\bvs\.?(?=\s)", re.IGNORECASE), "versus"),
    (re.compile(r"\bcf\.(?=\s)", re.IGNORECASE), "compare"),
    (re.compile(r"\bapprox\.(?=\s)", re.IGNORECASE), "approximately"),
    (re.compile(r"\bFigs?\.(?=\s*\d)"), "Figure"),
    (re.compile(r"\bet al\.(?=\s*[,;:)]|\s+[a-z(])"), "and colleagues"),
    (re.compile(r"\bet al\."), "and colleagues."),
    (re.compile(r"\bw/o\b"), "without"),
    (re.compile(r"\bn\.d\."), "no date"),
]


_SYMBOL_WORDS = {"●": "filled circle", "○": "open circle", "◐": "half circle", "✓": "check mark",
                 "✔": "check mark", "✗": "cross", "✘": "cross", "★": "star", "☆": "open star"}


def is_figure_placeholder(line: str) -> bool:
    return bool(_FIGURE_PLACEHOLDER_RE.match(line))


def _speak_url(m: re.Match) -> str:
    url = m.group(0).rstrip(".,;:!?")
    tail = m.group(0)[len(url):]
    host = re.sub(r"^(?:https?://)?", "", url, flags=re.IGNORECASE).split("/")[0].split("?")[0].split(":")[0]
    return (speak_domain(host) if host else "link") + tail


def _speak_code_id(m: re.Match) -> str:
    letters, num, year = m.group(1), m.group(2), m.group(3)
    spoken_num = digits_words(num) if num.startswith("0") else num
    out = f"{spell_letters(letters)} {spoken_num}"
    if year:
        out += f", {year_words(int(year))}"
    return out


def _speak_iso_date(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    return f"{_MONTHS[mo - 1]} {d}, {year_words(y)}"


def speakable(text: str, lang: str = "a") -> str:
    """Rewrite one paragraph so a TTS voice reads it like a person would."""
    t = text.translate({ord(c): None for c in _SUPERSCRIPTS})
    t = unicodedata.normalize("NFKC", t)
    t = _INVISIBLE_RE.sub("", t)
    t = BULLET_RE.sub("", t, count=1)
    t = _FOOTNOTE_BRACKET_RE.sub("", t)

    t = _EMAIL_RE.sub(lambda m: f"{m.group(1).replace('.', ' dot ')} at {speak_domain(m.group(2))}", t)
    t = _URL_RE.sub(_speak_url, t)
    t = _UUID_RE.sub("", t)
    t = _HEX_RE.sub("", t)
    t = _LONG_TOKEN_RE.sub("", t)

    english = lang in ("a", "b")
    if english:
        t = re.sub(r"\bU\+([0-9A-Fa-f]{4,6})\b", lambda m: "U plus " + " ".join(m.group(1).upper()), t)
        t = re.sub(r"(^|[·;:]\s*)[—–](?=\s)", r"\1dash", t)
        t = re.sub(r"\s+·\s+|\s*·\s*$", "; ", t)
        for sym, word in _SYMBOL_WORDS.items():
            t = re.sub(rf"\s*{re.escape(sym)}\s*", f" {word} ", t)
        t = _DOMAIN_RE.sub(lambda m: speak_domain(m.group(1)), t)
        t = _ISO_DATE_RE.sub(_speak_iso_date, t)
        t = _CVE_RE.sub(lambda m: f"C V E {year_words(int(m.group(1)))}, {m.group(2)}", t)
        t = _CODE_ID_RE.sub(_speak_code_id, t)
        for pat, rep in _ABBREVIATIONS:
            t = pat.sub(rep, t)
        t = re.sub(r"\bv(\d+(?:\.\d+)*)\b", r"version \1", t)
        t = re.sub(r"(?<!\w)#(\d+)", r"number \1", t)
        t = re.sub(r"(?<![\w.])~\s?(?=\d)", "about ", t)
        t = re.sub(r"\s&\s", " and ", t)
        t = re.sub(r"(\d)\s*–\s*(\d)", r"\1 to \2", t)
        t = t.replace("→", " to ").replace("≥", " at least ").replace("≤", " at most ")
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = re.sub(r"\s+\|\s+", ", ", t)
    t = t.replace("…", "...")

    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?)])", r"\1", t)
    t = re.sub(r"([(])\s+", r"\1", t)
    t = re.sub(r",\s*,+", ",", t)
    t = re.sub(r"^[,;:.\s]+", "", t)
    t = re.sub(r"([.!?;:]),", r"\1", t)
    return t.strip()


def ensure_sentence_end(text: str) -> str:
    """Headings and list fragments get a full stop so the voice pauses after them."""
    t = text.rstrip()
    if t and t[-1] not in ".!?:;\"'”’)":
        t += "."
    return t


# ---------------------------------------------------------------------------
# Sentence-aware chunking for the TTS model
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"'”’)]*\s+(?=[\"'“‘(]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def chunk_text(text: str, max_chars: int = 350) -> list[str]:
    """Group whole sentences into chunks of at most ``max_chars``.

    A single sentence longer than ``max_chars`` is split at clause boundaries
    (``;``, ``:``, ``,``) and only as a last resort at spaces.
    """
    chunks: list[str] = []
    cur = ""
    for sent in split_sentences(text):
        pieces = [sent] if len(sent) <= max_chars else _split_long(sent, max_chars)
        for p in pieces:
            if cur and len(cur) + 1 + len(p) > max_chars:
                chunks.append(cur)
                cur = p
            else:
                cur = f"{cur} {p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _split_long(sentence: str, max_chars: int) -> list[str]:
    for sep in ("; ", ": ", ", ", " "):
        parts = sentence.split(sep)
        if len(parts) == 1:
            continue
        out, cur = [], ""
        for i, part in enumerate(parts):
            piece = part + (sep.rstrip() if i < len(parts) - 1 else "")
            if cur and len(cur) + 1 + len(piece) > max_chars:
                out.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}" if cur else piece
        if cur:
            out.append(cur)
        if all(len(o) <= max_chars for o in out):
            return out
        sentence_parts = []
        for o in out:
            sentence_parts.extend([o] if len(o) <= max_chars else _split_long(o, max_chars))
        return sentence_parts
    return [sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)]
