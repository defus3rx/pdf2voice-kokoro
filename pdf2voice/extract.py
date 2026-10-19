"""PDF -> ordered sections of clean, speakable paragraphs (PyMuPDF based)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pymupdf

from . import textclean as tc

MARGIN_FRACTION = 0.10  # top/bottom band where running headers and footers live


@dataclass
class Row:
    page: int  # 0-based
    y0: float
    y1: float
    x0: float
    text: str
    size: float
    bold: bool
    kind: str = "text"  # "text" | "table"


@dataclass
class Para:
    kind: str  # "text" | "heading" | "table"
    text: str
    page: int
    size: float
    y0: float = 0.0


@dataclass
class Section:
    title: str
    page: int  # 0-based page where the section starts
    paras: list[Para] = field(default_factory=list)
    spoken: list[tuple[str, str]] = field(default_factory=list)  # (kind, text)

    @property
    def word_count(self) -> int:
        return sum(len(t.split()) for _, t in self.spoken)


@dataclass
class Document:
    path: Path
    title: str
    page_count: int
    pages: list[int]
    sections: list[Section]
    skipped: list[str]
    heading_source: str


# ---------------------------------------------------------------------------
# Page parsing
# ---------------------------------------------------------------------------

def parse_pages(spec: str | None, page_count: int) -> list[int]:
    """'10-55' / '1-5,8,20-' (1-based, inclusive) -> sorted 0-based page indices."""
    if not spec or spec.strip().lower() == "all":
        return list(range(page_count))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d*)\s*-\s*(\d*)|(\d+)", part)
        if not m:
            raise ValueError(f"Invalid page range: {part!r}")
        if m.group(3):
            a = b = int(m.group(3))
        else:
            a = int(m.group(1)) if m.group(1) else 1
            b = int(m.group(2)) if m.group(2) else page_count
        if a < 1 or b < a:
            raise ValueError(f"Invalid page range: {part!r}")
        pages.update(range(a - 1, min(b, page_count)))
    if not pages:
        raise ValueError(f"No pages selected by {spec!r} (document has {page_count} pages)")
    return sorted(pages)


def _span_is_bold(span: dict) -> bool:
    font = span.get("font", "")
    return bool(span["flags"] & 16) or bool(re.search(r"bold|black|heavy|semibold|demibold", font, re.I))


def _line_props(spans: list[dict]) -> tuple[float, bool]:
    chars = Counter()
    bold_chars = 0
    total = 0
    for s in spans:
        n = len(s["text"].strip())
        if not n:
            continue
        chars[round(s["size"] * 2) / 2] += n
        total += n
        if _span_is_bold(s):
            bold_chars += n
    size = chars.most_common(1)[0][0] if chars else 0.0
    return size, total > 0 and bold_chars / total > 0.6


def _is_footnote_marker(span: dict, body_hint: float | None) -> bool:
    t = span["text"].strip()
    if not t or not re.fullmatch(r"[\d,*†‡§¶]+", t):
        return False
    return bool(span["flags"] & 1)  # superscript


def page_lines(page: pymupdf.Page) -> list[dict]:
    """Text lines of a page in content order with geometry and font info."""
    out = []
    d = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if not _is_footnote_marker(s, None)]
            text = "".join(s["text"] for s in spans)
            if not text.strip():
                continue
            size, bold = _line_props(spans)
            x0, y0, x1, y1 = ln["bbox"]
            out.append({"text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "size": size, "bold": bold})
    return out


def merge_rows(lines: list[dict]) -> list[dict]:
    """Merge consecutive lines that sit on the same baseline (bullet + text, TOC title + page number)."""
    rows: list[dict] = []
    for ln in lines:
        if rows:
            prev = rows[-1]
            if abs(ln["y0"] - prev["y0"]) < 3 and ln["x0"] >= prev["x1"] - 1 and ln.get("table") == prev.get("table"):
                gap = ln["x0"] - prev["x1"]
                sep = "  " if gap > 15 else " "
                prev["text"] = prev["text"].rstrip() + sep + ln["text"].lstrip()
                prev["x1"] = ln["x1"]
                prev["y1"] = max(prev["y1"], ln["y1"])
                if len(ln["text"].strip()) > len(prev["text"].strip()) / 2:
                    prev["size"] = max(prev["size"], ln["size"]) if prev["bold"] == ln["bold"] else prev["size"]
                prev["bold"] = prev["bold"] and ln["bold"]
                continue
        rows.append(dict(ln))
    return rows


def _in_margin(row: dict, height: float) -> bool:
    return row["y1"] < height * MARGIN_FRACTION or row["y0"] > height * (1 - MARGIN_FRACTION)


def detect_running_text(doc: pymupdf.Document) -> set[str]:
    """Normalized header/footer lines repeated in the margins of most pages."""
    pages = []
    for page in doc:
        h = page.rect.height
        pages.append([r["text"] for r in merge_rows(page_lines(page)) if _in_margin(r, h)])
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({tc.normalize_for_repeat(l) for l in lines if l.strip()})
    threshold = max(2, int(len(pages) * 0.4 + 0.5))
    return {k for k, c in counts.items() if c >= threshold}


# ---------------------------------------------------------------------------
# Rows -> paragraphs
# ---------------------------------------------------------------------------

def extract_rows(doc: pymupdf.Document, pages: list[int], skip_tables: bool = True) -> tuple[list[Row], list[int]]:
    """All content rows of the selected pages, with tables collapsed into a single 'table' row.

    Returns the rows and the list of pages recognised as table-of-contents pages.
    """
    running = detect_running_text(doc)
    rows: list[Row] = []
    toc_pages: list[int] = []
    table_id = 0
    for pno in pages:
        page = doc[pno]
        h = page.rect.height
        lines = page_lines(page)

        body = [r for r in merge_rows(lines) if not _in_margin(r, h) or not (
            tc.normalize_for_repeat(r["text"]) in running or tc.is_page_number(r["text"]))]
        body = [r for r in body if not tc.is_page_number(r["text"]) or not _in_margin(r, h)]
        if tc.looks_like_toc_page([r["text"] for r in body]):
            toc_pages.append(pno)
            continue

        tables = []
        if skip_tables:
            try:
                tables = [pymupdf.Rect(t.bbox) + (-2, -2, 2, 2) for t in page.find_tables().tables]
            except Exception:
                tables = []
        ids = {}
        for i, _ in enumerate(tables):
            table_id += 1
            ids[i] = table_id
        for ln in lines:
            cx, cy = (ln["x0"] + ln["x1"]) / 2, (ln["y0"] + ln["y1"]) / 2
            ln["table"] = next((ids[i] for i, r in enumerate(tables) if r.contains(pymupdf.Point(cx, cy))), None)

        seen_tables: set[int] = set()
        for r in merge_rows(lines):
            text = r["text"].strip()
            if _in_margin(r, h) and (tc.normalize_for_repeat(text) in running or tc.is_page_number(text)):
                continue
            if r.get("table"):
                if r["table"] not in seen_tables:
                    seen_tables.add(r["table"])
                    rows.append(Row(pno, r["y0"], r["y1"], r["x0"], tc.TABLE_CUE, r["size"], False, "table"))
                continue
            if tc.is_figure_placeholder(text):
                continue
            kind = "table" if skip_tables and tc.is_table_row(text) else "text"
            rows.append(Row(pno, r["y0"], r["y1"], r["x0"], text if kind == "text" else tc.TABLE_CUE,
                            r["size"], r["bold"], kind))
    return rows, toc_pages


def body_font_size(rows: Iterable[Row]) -> float:
    c: Counter[float] = Counter()
    for r in rows:
        if r.kind == "text":
            c[r.size] += len(r.text)
    return c.most_common(1)[0][0] if c else 10.0


def _gap_ratio(a: Row, b: Row) -> float:
    """Vertical gap between consecutive rows in units of a's line height (inf across pages)."""
    if a.page != b.page:
        return float("inf")
    return (b.y0 - a.y1) / max(a.y1 - a.y0, 1.0)


def _heading_flags(rows: list[Row], body: float) -> list[bool]:
    """Large text is a heading; bold body-size text only when it stands alone as its own block."""
    flags = []
    for i, row in enumerate(rows):
        if row.kind != "text":
            flags.append(False)
            continue
        words = row.text.split()
        if row.size >= body * 1.15 and len(words) <= 25:
            flags.append(True)
            continue
        is_bold_head = (row.bold and len(words) <= 12 and row.text.rstrip()[-1:] not in ".,;"
                        and not tc.BULLET_RE.match(row.text))
        if is_bold_head:
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            prv = rows[i - 1] if i > 0 else None
            stands_alone = (nxt is None or nxt.kind != "text" or _gap_ratio(row, nxt) >= 0.6) and (
                prv is None or prv.kind != "text" or _gap_ratio(prv, row) >= 0.6)
            is_bold_head = stands_alone
        flags.append(is_bold_head)
    return flags


def _ends_sentence(text: str) -> bool:
    return text.rstrip()[-1:] in ".!?:\"”)" if text.strip() else True


def build_paragraphs(rows: list[Row], vocab: set[str] | None = None) -> tuple[list[Para], float]:
    body = body_font_size(rows)
    paras: list[Para] = []
    cur: Para | None = None
    prev: Row | None = None

    def flush():
        nonlocal cur
        if cur and cur.text.strip():
            paras.append(cur)
        cur = None

    heads = _heading_flags(rows, body)
    for row, heading in zip(rows, heads):
        if row.kind == "table":
            flush()
            if not paras or paras[-1].kind != "table":
                paras.append(Para("table", tc.TABLE_CUE, row.page, row.size, row.y0))
            prev = row
            continue

        bullet = bool(tc.BULLET_RE.match(row.text))
        if cur is not None and prev is not None and prev.kind == "text":
            same_page = row.page == prev.page
            line_h = max(prev.y1 - prev.y0, 1.0)
            gap = row.y0 - prev.y1
            if heading and cur.kind == "heading":
                if same_page and -line_h * 0.5 <= gap < line_h * 0.8 and abs(row.size - cur.size) < 1:
                    cur.text = tc.join_lines(cur.text, row.text, vocab)
                    prev = row
                    continue
            elif not heading and cur.kind == "text" and not bullet and abs(row.size - cur.size) < 1.5:
                if same_page and (gap < line_h * 0.6 or (gap < 0 and not _ends_sentence(cur.text))):
                    cur.text = tc.join_lines(cur.text, row.text, vocab)
                    prev = row
                    continue
                if not same_page and (not _ends_sentence(cur.text) or cur.text.rstrip().endswith("-")):
                    cur.text = tc.join_lines(cur.text, row.text, vocab)
                    prev = row
                    continue
        flush()
        cur = Para("heading" if heading else "text", row.text.strip(), row.page, row.size, row.y0)
        prev = row
    flush()
    # A paragraph that was split across a page by a table or heading-less page break can
    # still continue after a table cue; leave it split, the cue reads naturally.
    return paras, body


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _match_heading(paras: list[Para], title: str, page: int, start: int = 0) -> int | None:
    t = _norm(title)
    if not t:
        return None
    candidates = [i for i in range(start, len(paras)) if paras[i].page == page and paras[i].kind != "table"]
    for i in candidates:
        p = _norm(paras[i].text)
        if p and (p.startswith(t[:30]) or t.startswith(p[:30])) and len(p) >= min(len(t), 4):
            return i
    return None


def find_sections(doc: pymupdf.Document, paras: list[Para], body: float) -> tuple[list[tuple[int, str]], str]:
    """Return [(paragraph index, title)] for each top-level section and how they were found."""
    starts: list[tuple[int, str, float]] = []
    toc = doc.get_toc(simple=True)
    if toc:
        min_level = min(e[0] for e in toc)
        cursor = 0
        for level, title, pno in toc:
            if level != min_level or pno < 1:
                continue
            idx = _match_heading(paras, title, pno - 1, cursor)
            if idx is None:
                idx = next((i for i in range(cursor, len(paras)) if paras[i].page >= pno - 1), None)
                if idx is None or paras[idx].page > pno - 1 + 1:
                    continue
                size = 0.0
            else:
                size = paras[idx].size if paras[idx].kind == "heading" else 0.0
            starts.append((idx, re.sub(r"\s+", " ", title).strip(), size))
            cursor = idx
        sized = [s for _, _, s in starts if s]
        if len(sized) >= 2:
            top = max(sized)
            tiered = [(i, t) for i, t, s in starts if s >= top * 0.9]
            if len(tiered) >= 2:
                return _dedupe(tiered), "PDF outline (top-level headings by font size)"
        if len(starts) >= 2:
            return _dedupe([(i, t) for i, t, _ in starts]), "PDF outline"

    heads = [(i, p) for i, p in enumerate(paras) if p.kind == "heading" and p.size >= body * 1.4]
    if heads:
        sizes = Counter(round(p.size) for _, p in heads)
        frequent = [s for s, n in sizes.items() if n >= 2] or list(sizes)
        top = max(frequent)
        found = [(i, p.text) for i, p in heads if round(p.size) >= top - 0.5]
        if found:
            return _dedupe(found), "font-size heuristics"
    return [], "none (single section)"


def _dedupe(starts: list[tuple[int, str]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i, t in sorted(starts, key=lambda x: x[0]):
        if out and out[-1][0] == i:
            continue
        out.append((i, t))
    return out


def _drop_reference_lists(paras: list[Para]) -> list[Para]:
    """Remove 'References' / 'Bibliography' sub-lists inside a section."""
    out: list[Para] = []
    skipping_at: float | None = None
    for p in paras:
        if skipping_at is not None:
            if p.kind == "heading" and p.size >= skipping_at - 0.5:
                skipping_at = None
            else:
                continue
        if p.kind == "heading" and tc.REFERENCES_TITLE_RE.match(p.text):
            skipping_at = p.size
            continue
        out.append(p)
    return out


def _speak_section(sec: Section, lang: str) -> None:
    spoken: list[tuple[str, str]] = []
    for p in sec.paras:
        if p.kind == "table":
            if not spoken or spoken[-1][1] != tc.TABLE_CUE:
                spoken.append(("table", tc.TABLE_CUE))
            continue
        text = tc.speakable(p.text, lang)
        if not text or not re.search(r"[A-Za-z0-9]", text):
            continue
        spoken.append((p.kind, tc.ensure_sentence_end(text)))
    sec.spoken = spoken


def load_document(path: Path, pages_spec: str | None = None, keep_references: bool = False,
                  lang: str = "a") -> Document:
    doc = pymupdf.open(path)
    page_count = doc.page_count
    pages = parse_pages(pages_spec, page_count)
    rows, toc_pages = extract_rows(doc, pages)
    vocab = tc.vocab_from_text(r.text for r in rows if r.kind == "text")
    paras, body = build_paragraphs(rows, vocab)
    title = (doc.metadata or {}).get("title") or path.stem
    title = re.sub(r"\s+", " ", title).strip() or path.stem

    starts, source = find_sections(doc, paras, body)
    skipped: list[str] = []
    if toc_pages:
        skipped.append(f"table of contents (pages {', '.join(str(p + 1) for p in toc_pages)})")
    sections: list[Section] = []
    bounds = [i for i, _ in starts] + [len(paras)]
    if not starts or starts[0][0] > 0:
        first = bounds[0] if starts else len(paras)
        if first > 0:
            sections.append(Section("Front matter" if starts else title, paras[0].page, paras[:first]))
    for k, (i, t) in enumerate(starts):
        sections.append(Section(t, paras[i].page, paras[i:bounds[k + 1]]))

    kept: list[Section] = []
    for sec in sections:
        if tc.TOC_TITLE_RE.match(sec.title):
            skipped.append(f"section '{sec.title}' (table of contents)")
            continue
        if not keep_references and tc.REFERENCES_TITLE_RE.match(sec.title):
            skipped.append(f"section '{sec.title}' (reference list; use --keep-references to include)")
            continue
        if not keep_references:
            sec.paras = _drop_reference_lists(sec.paras)
        _speak_section(sec, lang)
        if sum(k == "text" for k, _ in sec.spoken) == 0:
            if sec.spoken:
                skipped.append(f"section '{sec.title}' (no readable text, only figures or tables)")
            continue
        kept.append(sec)
    doc.close()
    return Document(path, title, page_count, pages, kept, skipped, source)
