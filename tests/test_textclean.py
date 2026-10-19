from pdf2voice.textclean import (
    TABLE_CUE,
    chunk_text,
    ensure_sentence_end,
    find_repeated_lines,
    is_page_number,
    is_table_row,
    is_toc_line,
    join_hyphenated,
    join_lines,
    looks_like_toc_page,
    reflow,
    remove_headers_footers,
    skip_tables,
    speakable,
    year_words,
)


# --- hyphen joining ---------------------------------------------------------

def test_join_soft_hyphen():
    assert join_hyphenated("an exam-\nple of text") == "an example of text"


def test_keep_real_compound_hyphen():
    assert join_hyphenated("indirect prompt-\ninjection attacks") == "indirect prompt-injection attacks"


def test_hyphen_before_capital_or_digit_is_kept():
    assert join_hyphenated("COVID-\n19 cases") == "COVID-19 cases"
    assert join_hyphenated("Anti-\nVirus") == "Anti-Virus"


def test_document_vocab_decides_unknown_words():
    # Neither "frobni" nor "cation" are words, but the document uses "frobnication" elsewhere.
    assert join_hyphenated("frobni-\ncation", vocab={"frobnication"}) == "frobnication"


def test_reflow_joins_lines_and_keeps_paragraph_breaks():
    lines = ["This is the first line of a para-", "graph that wraps.", "", "Second paragraph", "continues here."]
    assert reflow(lines) == ["This is the first line of a paragraph that wraps.", "Second paragraph continues here."]


def test_reflow_bullets_start_new_paragraphs():
    lines = ["Outcomes include:", "• Disclosure of", "secrets.", "• Manipulation of output."]
    assert reflow(lines) == ["Outcomes include:", "• Disclosure of secrets.", "• Manipulation of output."]


def test_wrapped_url_is_not_split_with_a_space():
    assert join_lines("see https://example.com/blog/", "post-name/ for more") == "see https://example.com/blog/post-name/ for more"


# --- headers, footers and page numbers --------------------------------------

def test_page_numbers():
    for s in ["12", "Page 12", "page 3 of 80", "xii", "- 4 -", "7/120"]:
        assert is_page_number(s), s
    for s in ["12 attacks were observed", "Page layout", "did", "2023-08-01"]:
        assert not is_page_number(s), s


def test_header_footer_removal():
    topics = ["injection", "disclosure", "agency", "supply chains", "poisoning", "consumption"]
    pages = [
        ["ACME Security Report", f"This page is about {t}.", f"Attackers abuse {t} in practice.", "genai.example.org", f"Page {i}"]
        for i, t in enumerate(topics, start=1)
    ]
    repeated = find_repeated_lines(pages)
    assert {"acme security report", "genai.example.org", "page #"} <= repeated
    cleaned = remove_headers_footers(pages)
    for t, lines in zip(topics, cleaned):
        assert lines == [f"This page is about {t}.", f"Attackers abuse {t} in practice."]


def test_lone_page_number_removed_even_if_not_repeated():
    pages = [["Intro", "Some text."], ["Real content here.", "17"], ["Other content.", "More."]]
    cleaned = remove_headers_footers(pages)
    assert cleaned[1] == ["Real content here."]


def test_header_removal_needs_several_pages():
    assert find_repeated_lines([["Title", "x"], ["Title", "y"]]) == set()


# --- table of contents -------------------------------------------------------

def test_toc_lines():
    assert is_toc_line("Introduction ........................ 5")
    assert is_toc_line("2.1 Threat Model . . . . . . 14")
    assert is_toc_line("LLM01:2026 Prompt Injection    10")
    assert not is_toc_line("We pulled together a corpus of 7,714 real incidents.")
    assert not is_toc_line("The incident count rose sharply in 2024")


def test_toc_page_detection():
    toc = ["Table of Content", "Letter from the Project Leads 5", "What's New 6", "LLM01:2026 Prompt Injection 10",
           "LLM02:2026 Sensitive Information Disclosure 18"]
    assert looks_like_toc_page(toc)
    body = ["Stop trying to build a model that cannot be fooled.", "Build the system around it, so that when it",
            "is fooled, nothing important breaks.", "This year we tested the vote against 2024"]
    assert not looks_like_toc_page(body)


# --- tables ------------------------------------------------------------------

def test_symbol_rows_are_table_rows():
    assert is_table_row("LLM01 ● ○ ● ● — ✓")
    assert is_table_row("● ●")
    assert not is_table_row("● Disclosure of sensitive information to an attacker.")
    assert not is_table_row("The model — not the user — decides.")


def test_skip_tables_replaces_each_table_with_one_cue():
    lines = [
        "The coverage matrix is shown below.",
        "Risk ASI DSGAI ATLAS ATT&CK CWE RMF",
        "LLM01 ● ● ○ ● ● ○",
        "LLM02 ● ○ ○ — ● ●",
        "Unbounded",
        "Consumption",
        "LLM06 ● ● ● ○ ● ●",
        "Each entry is discussed in detail.",
        "LLM10 ✓ ✓ ✗ ✓",
    ]
    assert skip_tables(lines) == [
        "The coverage matrix is shown below.",
        TABLE_CUE,
        "Each entry is discussed in detail.",
        TABLE_CUE,
    ]


# --- speakable text -----------------------------------------------------------

def test_abbreviations():
    assert speakable("Use filters, e.g. regex.") == "Use filters, for example, regex."
    assert speakable("The model, i.e. the LLM, decides.") == "The model, that is, the L L M, decides." or \
        speakable("The model, i.e. the LLM, decides.") == "The model, that is, the LLM, decides."


def test_owasp_ids():
    assert speakable("LLM01:2026 Prompt Injection") == "L L M zero one, twenty twenty-six Prompt Injection"
    assert speakable("see LLM10 and ASI02") == "see L L M 10 and A S I zero two"


def test_urls_emails_and_hashes():
    assert speakable("Link: https://creativecommons.org/licenses/by-sa/4.0/legalcode.") == "Link: creativecommons dot org."
    assert speakable("Visit genai.owasp.org today") == "Visit genai dot owasp dot org today"
    assert speakable("Mail jane.doe@example.com now") == "Mail jane dot doe at example dot com now"
    assert speakable("commit 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b fixed it") == "commit fixed it"


def test_bullets_and_footnotes_removed():
    assert speakable("• Disclosure of secrets.[12]") == "Disclosure of secrets."
    assert speakable("as shown¹ before") == "as shown before"


def test_dates_cves_and_misc():
    assert speakable("released 2025-12-09") == "released December 9, twenty twenty-five"
    assert speakable("CVE-2025-53773 was fixed") == "C V E twenty twenty-five, 53773 was fixed"
    assert speakable("Scenario #3") == "Scenario number 3"
    assert speakable("NIST AI 600-1") == "NIST AI 600-1"


def test_citations_codepoints_and_legends():
    assert speakable("(Clusmann et al., 2025)") == "(Clusmann and colleagues, 2025)"
    assert speakable("strip U+200B characters") == "strip U plus 2 0 0 B characters"
    assert speakable("Legend: ● primary · ○ supporting · — none") == \
        "Legend: filled circle primary; open circle supporting; dash none"
    assert speakable("Zenity | Founder") == "Zenity, Founder"


def test_numeric_hyphen_wrap_joins_without_space():
    assert join_lines("v1.0 (2026-03-", "17)") == "v1.0 (2026-03-17)"


def test_year_words():
    assert year_words(2026) == "twenty twenty-six"
    assert year_words(2005) == "two thousand five"
    assert year_words(1999) == "nineteen ninety-nine"


def test_sentence_end():
    assert ensure_sentence_end("Description") == "Description."
    assert ensure_sentence_end("Done!") == "Done!"


# --- chunking -----------------------------------------------------------------

def test_chunks_never_cut_mid_sentence():
    text = " ".join(f"Sentence number {i} is here and it is fine." for i in range(40))
    chunks = chunk_text(text, max_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert " ".join(chunks) == text


def test_long_sentence_split_at_clauses():
    sent = ", ".join(["a clause with several words"] * 30) + "."
    chunks = chunk_text(sent, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks) == sent
