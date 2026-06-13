"""
ECB Document Collection Pipeline  —  v2
========================================
Augmenting Volatility Models with Text-Based News Analytics
Research Paper — Data Collection Only (no scoring)

This script collects and cleans ECB communication documents for downstream
sentiment / stance scoring. All text cleaning happens HERE, at collection
time, so that the scoring pipeline can simply read the saved body text and
operate on it directly with no further processing.

WHAT'S DIFFERENT FROM THE PREVIOUS COLLECTION SCRIPT
-----------------------------------------------------
1. ADDS PRE-2015 MONTHLY BULLETINS (2008-2014). The ECB published a Monthly
   Bulletin every month from 1999 to December 2014, when it was replaced
   by the Economic Bulletin. The previous collection only had the post-2015
   Economic Bulletins — so our in-sample period (2008-2018) had a coverage
   gap from 2008 through 2014 for analytical/outlook content. We close this
   by directly downloading the Monthly Bulletin PDFs from the predictable URL:
       https://www.ecb.europa.eu/pub/pdf/mobu/mb{YYYY}{MM}en.pdf

2. STRIPS Q&A SECTIONS from MPS and economic bulletin documents. The previous
   collection saved full press conference transcripts including journalist
   questions. For sentiment scoring this was reasonable (questions can reflect
   market mood). For stance scoring it's noise — journalists are not the ECB
   voice. We cut at the first Q&A start marker and save only the prepared
   statement portion.

3. STRIPS PDF BOILERPLATE PREFIXES. PDF documents from the ECB website have
   header artifacts at the start of the body (website URL, copyright, postal
   address, speaker name and title) that are not actual ECB content. We
   detect these and skip past them.

4. DROPS DUPLICATES. From 2021 onward, every MPS event has TWO foedb entries:
   one as the regular `monetary_policy_statement` (HTML, includes Q&A) and
   one as `combined_monetary_policy_statement` (PDF, statement-only). They
   cover the same event. We drop the combined version and keep the regular.
   17 dates are affected.

5. DROPS KNOWN SLIDE DECKS. Three "speeches" in the foedb are actually slide
   decks for panel discussions. They have different language structure
   (bullet points, slide titles) and shouldn't be in a stance corpus.
   Doc IDs: 37829, 35711, 35390.

6. WARNS ABOUT NEW SUSPECTED SLIDE DECKS. The script flags any document
   whose body starts with patterns like "background slides" or "slides to"
   so we can review them rather than silently include them.

OUTPUT STRUCTURE
----------------
    output/
      ecb_documents_master.csv           — clean master CSV
      ecb_raw_texts_clean/               — one .txt per document, cleaned body
        {year}_{doc_type}_{raw_id}.txt
      collection.log

USAGE
-----
    python ecb_sentiment_pipeline_v2.py

The script is incremental: if .txt files already exist for documents in the
old `ecb_raw_texts/` directory, we re-process them (apply Q&A and boilerplate
stripping) and save to `ecb_raw_texts_clean/`. Documents that need fresh
download (e.g., monthly bulletins) are downloaded from scratch.
"""

import os
import re
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from io import BytesIO

from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("WARNING: pypdf not installed. PDF documents won't be processed.")
    print("Install with: pip install pypdf")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

START_YEAR = 2006
END_YEAR   = 2023

# foedb document type codes
TYPE_CODES = {
    19:  "speech",
    54:  "monetary_policy_statement",
    6:   "economic_bulletin",
    291: "combined_monetary_policy_statement",
    407: "governing_council_statement",
}

# Speaker filter — applied ONLY to speeches (type 19)
TARGET_SPEAKERS = {
    "Jean-Claude Trichet",   # President –Oct 2011
    "Mario Draghi",          # President Nov 2011–Oct 2019
    "Christine Lagarde",     # President Nov 2019–
    "Jürgen Stark",          # Chief Economist –Dec 2011
    "Peter Praet",           # Chief Economist Jun 2012–May 2019
    "Philip R. Lane",        # Chief Economist Jun 2019–
}

# Pre-2015 monthly bulletins (added 2008-2014)
MONTHLY_BULLETIN_START_YEAR = 2008
MONTHLY_BULLETIN_END_YEAR   = 2014

# Documents to drop entirely (known slide decks identified during cleanup).
# IDs 21471, 24949, 25100 are Philip Lane slide-deck PDFs (Strategy Review,
# OMFIF panel, Future of the Euro) — bullet-point slides, no prose.
KNOWN_SLIDE_DECK_IDS = {37829, 35711, 35390, 21471, 24949, 25100}

# Patterns that indicate a slide-deck-style document (we warn but don't auto-drop)
SLIDE_DECK_PATTERNS = [
    r"\bbackground slides\b",
    r"\bslides? to\b",
    r"\bpanel discussion\b",
    r"\bpresentation slides\b",
]
SLIDE_DECK_REGEX = re.compile("|".join(SLIDE_DECK_PATTERNS), re.IGNORECASE)

# Q&A start markers — separated by case sensitivity to avoid two known bugs:
#
#   1. The phrase "Jump to the transcript of the questions and answers" appears
#      in ECB press-conference HTML pages as a top-of-page NAVIGATION LINK,
#      not as the actual Q&A section divider. Matching it (as the previous
#      regex did) chopped the entire introductory statement from all 182 MPS
#      documents. We replaced it with the canonical ECB transition phrase
#      "We are now at your disposal for [your] questions" (or close variants),
#      which marks the actual end of the introductory statement in 178/182
#      MPS docs. Verified by inspection of the legacy fomc-scored corpus.
#
#   2. The catch-all `\bQuestion[:.] ` was IGNORECASE, so it matched natural
#      prose use of "question:" (e.g., "the question: how should we balance ...").
#      The case-sensitive version still catches the "Question:" section header
#      while leaving prose alone.
QA_START_PATTERNS_CI = [
    # Canonical ECB Q&A transition phrases (case-insensitive)
    r"(?:We|I)\s+(?:are|am)\s+now\s+(?:at\s+your\s+disposal|ready\s+to\s+take|"
    r"happy\s+to\s+take|here\s+to\s+take|happy|ready|glad)\b",
    # Older/alternative transcript header
    r"Transcript of the questions asked",
    # Newline-anchored Question section header (rare in HTML-extracted text)
    r"\n\s*\*?\*?\s*Question\s*\*?\*?\s*[:.\-]",
]
QA_START_PATTERNS_CS = [
    # Catch-all Question header — must be capitalized "Question" followed by
    # ":" or "." then a capital letter (start of journalist question).
    r"(?:^|[\s.,])Question\s*[:.]\s+[A-Z]",
]
QA_REGEX_CI = re.compile("|".join(QA_START_PATTERNS_CI), re.IGNORECASE)
QA_REGEX_CS = re.compile("|".join(QA_START_PATTERNS_CS))

# PDF boilerplate prefix patterns. We scan the first 1500 chars of the body
# for these and cut past them. Each pattern represents a self-contained
# administrative chunk that ECB PDFs prepend.
#
# These patterns are anchored to phrases that are uniquely administrative
# (postal addresses, copyright notices) and end at a semantically clear
# boundary so we don't accidentally strip real content.
PDF_BOILERPLATE_PATTERNS = [
    # Website URL and copyright character together — common signature in
    # speech PDFs. e.g. "[www.ecb.europa.eu](http://www.ecb.europa.eu) © ..."
    r"\[www\.ecb\.europa\.eu\]\(http://www\.ecb\.europa\.eu\)\s*©\s*",

    # Combined MPS PDFs: postal address block ending at the
    # "Reproduction is permitted..." sentence.
    r"European Central Bank\s+Directorate General Communications.*?"
    r"Reproduction is permitted provided that the source is acknowledged\.\s*",

    # Older Monthly Bulletin format with full Kaiserstrasse address.
    # Matches from "© European Central Bank YYYY" through the entire
    # address/contact block, ending at "http://www.ecb.europa.eu" (the
    # last URL in the boilerplate).
    r"©\s*European Central Bank\s+\d{4}\s+"
    r"(?:All rights reserved.*?)?"  # optional copyright sentence
    r"Address\s+Kaiserstrasse.*?"
    r"(?:Website|Internet)\s+https?://(?:www\.)?ecb\.(?:int|europa\.eu)",
]
PDF_BOILERPLATE_REGEX = re.compile(
    "(?:" + "|".join(PDF_BOILERPLATE_PATTERNS) + ")",
    re.IGNORECASE | re.DOTALL,
)

# Monthly Bulletin front-matter end-anchor. Every Monthly Bulletin (2008-2014)
# ends its front matter (publication imprint + table of contents +
# country-code abbreviations) with this exact phrase, immediately before the
# real content begins (page header + EDITORIAL). Verified on all 84 bulletins
# in the corpus. We strip everything from the start of the body through the
# end of this match (within the first 6000 chars only, to avoid pathological
# false matches deep in the document).
MONTHLY_BULLETIN_FRONTMATTER_END_REGEX = re.compile(
    r"In\s+accordance\s+with\s+(?:Community|EU)\s+practice,\s+the\s+EU\s+countries\s+are\s+listed\s+"
    r"in\s+this\s+Bulletin\s+using\s+the\s+alphabetical\s+order\s+of\s+the\s+country\s+names\s+"
    r"in\s+the\s+national\s+languages\.",
    re.IGNORECASE,
)
MONTHLY_BULLETIN_FRONTMATTER_SCAN_CHARS = 6000

OUTPUT_DIR     = Path("./output")
RAW_TEXT_DIR   = OUTPUT_DIR / "ecb_raw_texts_clean"   # NEW location, leave old one alone
OLD_RAW_DIR    = OUTPUT_DIR / "ecb_raw_texts"         # for incremental reprocessing
MASTER_CSV     = OUTPUT_DIR / "ecb_documents_master.csv"

REQUEST_DELAY = 0.4

ECB_BASE   = "https://www.ecb.europa.eu"
FOEDB_BASE = f"{ECB_BASE}/foedb/dbs/foedb/publications.en"

USER_AGENT = "Mozilla/5.0 (academic research; graduate dissertation project)"


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
RAW_TEXT_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FOEDB API — collection
# ─────────────────────────────────────────────────────────────────────────────

def get_api_version():
    url = f"{FOEDB_BASE}/versions.json"
    log.info(f"Fetching API version from {url}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    entry = data[0]
    version_id = entry.get("version") or entry.get("versionId")
    hash_key   = entry.get("hash")    or entry.get("hashKey")
    log.info(f"  API version: {version_id} / {hash_key}")
    return version_id, hash_key


def download_all_records(version_id, hash_key):
    """Download all chunk files and return flat record arrays."""
    FIELDS = 13
    base = f"{FOEDB_BASE}/{version_id}/{hash_key}"

    meta_url = f"{base}/metadata.json"
    log.info(f"Fetching metadata from {meta_url}")
    meta = requests.get(meta_url, timeout=15).json()
    total_records = meta["total_records"]
    chunk_size    = meta["chunk_size"]
    n_chunks      = (total_records + chunk_size - 1) // chunk_size
    log.info(f"  Total records: {total_records:,}, chunks: {n_chunks}")

    records = []
    for i in tqdm(range(n_chunks), desc="Downloading API chunks"):
        url = f"{base}/data/0/chunk_{i}.json"
        try:
            flat = requests.get(url, timeout=15).json()
            for j in range(0, len(flat), FIELDS):
                rec = flat[j:j + FIELDS]
                if len(rec) == FIELDS:
                    records.append(rec)
        except Exception as e:
            log.warning(f"  Chunk {i} failed: {e}")
        time.sleep(REQUEST_DELAY)

    log.info(f"  Total records parsed: {len(records):,}")
    return records


def parse_record(r):
    if not isinstance(r, list) or len(r) < 10:
        return None
    url_paths = r[9]
    if isinstance(url_paths, list) and url_paths:
        en_urls = [u for u in url_paths if isinstance(u, str) and u.endswith(".en.html")]
        url_path = en_urls[0] if en_urls else (url_paths[0] if isinstance(url_paths[0], str) else None)
    else:
        url_path = None
    return {
        "raw_id":        r[0],
        "pub_timestamp": r[1],
        "year":          r[2],
        "type_code":     r[4],
        "boardmember":   r[7] or "",
        "url_path":      url_path,
    }


def filter_records(records):
    """Apply type / year / speaker filters; drop known slide decks."""
    filtered = []
    skipped = {"type": 0, "year": 0, "speaker": 0, "slide_deck": 0, "no_url": 0}

    for raw in records:
        rec = parse_record(raw)
        if rec is None:
            continue

        if rec["type_code"] not in TYPE_CODES:
            skipped["type"] += 1
            continue

        if not (START_YEAR <= rec["year"] <= END_YEAR):
            skipped["year"] += 1
            continue

        if not rec["url_path"]:
            skipped["no_url"] += 1
            continue

        # Speaker filter on speeches only
        if rec["type_code"] == 19:
            speakers = set(s.strip() for s in rec["boardmember"].split("|") if s.strip())
            if not (speakers & TARGET_SPEAKERS):
                skipped["speaker"] += 1
                continue

        # Drop known slide decks
        if rec["raw_id"] in KNOWN_SLIDE_DECK_IDS:
            skipped["slide_deck"] += 1
            continue

        filtered.append({
            "raw_id":    rec["raw_id"],
            "year":      rec["year"],
            "type_code": rec["type_code"],
            "doc_type":  TYPE_CODES[rec["type_code"]],
            "speaker":   rec["boardmember"],
            "url":       ECB_BASE + rec["url_path"],
        })

    log.info(f"  Filtered to {len(filtered)} documents")
    log.info(f"  Skipped: {dict(skipped)}")
    counts = Counter(r["doc_type"] for r in filtered)
    log.info(f"  By type: {dict(counts)}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEDUPLICATION — combined MPS overlapping with regular MPS
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_combined_mps(docs):
    """
    Drop `combined_monetary_policy_statement` documents whose date already
    has a regular `monetary_policy_statement`. The combined version is the
    statement-only PDF; the regular version is the full HTML press conference
    page. They cover the same event.

    To compare dates, we use the URL-encoded date pattern (ecb.is{YYMMDD} for
    regular MPS and ecb.ds{YYMMDD} for combined MPS).
    """
    def extract_date_key(url):
        m = re.search(r'ecb\.(?:is|ds)(\d{6})', url)
        return m.group(1) if m else None

    regular_mps_dates = {
        extract_date_key(d["url"])
        for d in docs
        if d["doc_type"] == "monetary_policy_statement"
    }
    regular_mps_dates.discard(None)

    out = []
    n_dropped = 0
    for d in docs:
        if d["doc_type"] == "combined_monetary_policy_statement":
            date_key = extract_date_key(d["url"])
            if date_key in regular_mps_dates:
                n_dropped += 1
                continue
        out.append(d)

    log.info(f"  Dropped {n_dropped} combined MPS duplicates of regular MPS dates")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. MONTHLY BULLETINS (pre-2015) — direct URL construction
# ─────────────────────────────────────────────────────────────────────────────

def monthly_bulletin_url(year, month):
    """Build the ECB Monthly Bulletin PDF URL for a given year-month."""
    return f"https://www.ecb.europa.eu/pub/pdf/mobu/mb{year}{month:02d}en.pdf"


def collect_monthly_bulletins():
    """
    Build records for the pre-2015 Monthly Bulletins, one per (year, month)
    in [2008, 2014]. We construct synthetic raw_ids in a high range that
    won't collide with foedb IDs (which are 5-digit integers — we use
    900000+ for bulletins to be safe).
    """
    bulletins = []
    for year in range(MONTHLY_BULLETIN_START_YEAR, MONTHLY_BULLETIN_END_YEAR + 1):
        for month in range(1, 13):
            url = monthly_bulletin_url(year, month)
            # Synthetic ID: 900YYMM (e.g., 900801 = Jan 2008)
            synthetic_id = 900000 + (year - 2000) * 100 + month
            bulletins.append({
                "raw_id":    synthetic_id,
                "year":      year,
                "type_code": -1,  # synthetic
                "doc_type":  "monthly_bulletin",
                "speaker":   "",
                "url":       url,
            })
    log.info(f"  Constructed {len(bulletins)} monthly bulletin records ({MONTHLY_BULLETIN_START_YEAR}–{MONTHLY_BULLETIN_END_YEAR})")
    return bulletins


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEXT SCRAPING — HTML and PDF
# ─────────────────────────────────────────────────────────────────────────────

def extract_pub_date_from_url(url):
    """Extract publication date from ECB URL filename pattern."""
    # speeches, MPS, bulletins, opinions, combined MPS
    m = re.search(r"ecb\.(?:sp|is|eb|op|ds)(\d{2})(\d{2})(\d{2})", url)
    if m:
        yy, mm, dd = m.groups()
        try:
            return str(datetime.strptime(f"20{yy}-{mm}-{dd}", "%Y-%m-%d").date())
        except ValueError:
            pass
    # Pre-2015 monthly bulletin: mb{YYYY}{MM}en.pdf — first day of month
    m = re.search(r"mb(\d{4})(\d{2})en\.pdf", url)
    if m:
        yyyy, mm = m.groups()
        try:
            return str(datetime.strptime(f"{yyyy}-{mm}-01", "%Y-%m-%d").date())
        except ValueError:
            pass
    return None


def extract_pub_date_from_html(soup, url):
    """Try multiple sources to extract pub_date from an ECB page."""
    # 1. HTML metadata tags
    for meta in soup.find_all("meta"):
        try:
            name = (meta.get("name") or meta.get("property") or "").lower()
            content = meta.get("content") or ""
            if name in ("article:published_time", "dc.date", "date") and content:
                return str(datetime.strptime(content[:10], "%Y-%m-%d").date())
        except (ValueError, AttributeError):
            continue

    # 2. URL filename pattern
    pub_date = extract_pub_date_from_url(url)
    if pub_date:
        return pub_date

    # 3. First date string in body text
    try:
        text_snippet = soup.get_text()[:2000]
        m = re.search(
            r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d{4})\b",
            text_snippet,
        )
        if m:
            return str(datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y"
            ).date())
    except (ValueError, AttributeError):
        pass

    return None


def strip_qa(text):
    """
    Remove Q&A section from document body. Cuts at the EARLIEST match of
    any QA start marker (across both case-insensitive and case-sensitive
    pattern sets). Returns (stripped_text, was_stripped, n_chars_removed).
    Speeches typically have no Q&A and pass through unchanged.
    """
    m_ci = QA_REGEX_CI.search(text)
    m_cs = QA_REGEX_CS.search(text)
    candidates = [m for m in (m_ci, m_cs) if m is not None]
    if not candidates:
        return text, False, 0
    earliest = min(candidates, key=lambda m: m.start())
    stripped = text[:earliest.start()].strip()
    return stripped, True, len(text) - len(stripped)


def strip_pdf_boilerplate(text):
    """
    Strip leading administrative boilerplate from PDF-extracted text.

    PDF documents from the ECB website often start with website URL,
    copyright character, postal address, etc. before the actual content.
    We scan only the first 1500 chars and drop everything matched by our
    boilerplate patterns.

    Returns (stripped_text, was_stripped).
    """
    # Only scan the first 1500 chars — we don't want to accidentally
    # strip mid-document text that happens to match the patterns.
    head = text[:1500]
    matches = list(PDF_BOILERPLATE_REGEX.finditer(head))
    if not matches:
        return text, False
    last_match = matches[-1]
    cut_at = last_match.end()
    return text[cut_at:].strip(), True


def strip_monthly_bulletin_frontmatter(text):
    """
    Strip Monthly Bulletin publication imprint, table of contents, and
    country-code abbreviation list. Anchored on the unique end-of-front-matter
    phrase that appears in every 2008-2014 Monthly Bulletin immediately
    before content. Returns (stripped_text, was_stripped).
    """
    head = text[:MONTHLY_BULLETIN_FRONTMATTER_SCAN_CHARS]
    m = MONTHLY_BULLETIN_FRONTMATTER_END_REGEX.search(head)
    if not m:
        return text, False
    return text[m.end():].lstrip(), True


def looks_like_slide_deck(text):
    """Quick heuristic: does the body start with slide-deck signatures?"""
    head = text[:800].lower()
    return SLIDE_DECK_REGEX.search(head) is not None


def scrape_html(url):
    """Fetch and extract clean body text from an HTML ECB page."""
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pub_date = extract_pub_date_from_html(soup, url)

    main = soup.find("main") or soup.find("div", id="main")
    if not main:
        log.warning(f"  No <main> for {url}")
        return pub_date, None

    exclude_classes = {"related-topics", "related-publications",
                      "address-box", "sharing", "feedback", "tablet-only"}
    parts = []
    for div in main.find_all("div", recursive=False):
        try:
            classes = div.get("class") or []
            if isinstance(classes, str):
                classes = classes.split()
            if any(c in exclude_classes for c in classes):
                continue
            parts.append(div)
        except (AttributeError, TypeError):
            continue

    if parts:
        text = " ".join(p.get_text(separator=" ", strip=True) for p in parts)
    else:
        text = main.get_text(separator=" ", strip=True)

    text = re.sub(r"\s+", " ", text).strip()
    # Strip the universal "CONTACT European Central Bank..." footer
    text = re.sub(
        r"CONTACT\s+European Central Bank.*$", "",
        text, flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    return pub_date, (text if text else None)


def scrape_pdf(url):
    """Fetch and extract text from a PDF; strip boilerplate prefix."""
    if not PYPDF_AVAILABLE:
        return None, None
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=30, stream=True)
    r.raise_for_status()
    reader = PdfReader(BytesIO(r.content))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pass
    text = " ".join(pages)
    text = re.sub(r"\s+", " ", text).strip()

    pub_date = extract_pub_date_from_url(url)

    if not text:
        return pub_date, None
    return pub_date, text


def scrape_document(url):
    """
    Top-level scraper. Returns (pub_date, raw_text) where raw_text is the
    body without metadata header but BEFORE Q&A or boilerplate stripping.
    """
    try:
        if url.lower().endswith(".pdf"):
            return scrape_pdf(url)
        return scrape_html(url)
    except Exception as e:
        log.warning(f"  Scrape failed for {url}: {e}")
        return None, None


def clean_document_text(raw_text, is_pdf, doc_type=None):
    """
    Apply text cleaning: strip PDF boilerplate (if PDF), strip Monthly
    Bulletin front matter (if doc_type == "monthly_bulletin"), then strip Q&A.
    Returns (clean_text, dict_of_flags).
    """
    flags = {"qa_stripped": False, "qa_chars_removed": 0,
            "pdf_boilerplate_stripped": False,
            "monthly_bulletin_frontmatter_stripped": False,
            "looks_like_slide_deck": False}
    text = raw_text

    if is_pdf:
        text, was_stripped = strip_pdf_boilerplate(text)
        flags["pdf_boilerplate_stripped"] = was_stripped

        # After boilerplate stripping, check if what remains looks like slides
        if looks_like_slide_deck(text):
            flags["looks_like_slide_deck"] = True

    if doc_type == "monthly_bulletin":
        text, was_mb = strip_monthly_bulletin_frontmatter(text)
        flags["monthly_bulletin_frontmatter_stripped"] = was_mb

    text, was_qa, n_qa = strip_qa(text)
    flags["qa_stripped"] = was_qa
    flags["qa_chars_removed"] = n_qa

    return text, flags


# ─────────────────────────────────────────────────────────────────────────────
# 5. INCREMENTAL TEXT FILE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def read_old_raw_text(text_path):
    """Read body text from existing raw_text file (skip metadata header)."""
    with open(text_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    parts = re.split(r"\n\s*\n", content, maxsplit=1)
    return parts[1] if len(parts) == 2 else content


def write_clean_text_file(out_path, doc, body_text):
    """Save a cleaned text file with metadata header."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"PUB_DATE: {doc.get('pub_date')}\n")
        f.write(f"URL: {doc.get('url')}\n")
        f.write(f"TYPE: {doc.get('doc_type')}\n")
        f.write(f"SPEAKER: {doc.get('speaker')}\n\n")
        f.write(body_text)


def process_documents(docs):
    """
    For each document:
      - If a clean .txt file already exists at the new path, skip
      - Else if old .txt exists at OLD_RAW_DIR, read body, clean, save to NEW
      - Else download from URL, clean, save to NEW

    Updates each doc dict with text_path, pub_date, and cleaning flags.
    Returns the updated list along with summary statistics.
    """
    stats = {
        "already_clean": 0,
        "reprocessed_from_old": 0,
        "freshly_downloaded": 0,
        "failed": 0,
        "qa_stripped": 0,
        "pdf_boilerplate_stripped": 0,
        "monthly_bulletin_frontmatter_stripped": 0,
        "suspected_slide_decks": [],
        "total_qa_chars_removed": 0,
    }

    for doc in tqdm(docs, desc="Processing documents"):
        safe_id  = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(doc["raw_id"]))
        out_path = RAW_TEXT_DIR / f"{doc['year']}_{doc['doc_type']}_{safe_id}.txt"
        old_path = OLD_RAW_DIR  / f"{doc['year']}_{doc['doc_type']}_{safe_id}.txt"

        doc["text_path"] = str(out_path)
        if "pub_date" not in doc:
            doc["pub_date"] = None

        # Already cleaned in this run?
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                m = re.match(r"PUB_DATE: (.+)", f.readline())
                if m:
                    doc["pub_date"] = m.group(1).strip()
            stats["already_clean"] += 1
            continue

        is_pdf = doc["url"].lower().endswith(".pdf")
        raw_text = None

        # Try to reprocess from existing old file first (avoids re-download)
        if old_path.exists():
            raw_text = read_old_raw_text(old_path)
            doc["pub_date"] = extract_pub_date_from_url(doc["url"])
            stats["reprocessed_from_old"] += 1
        else:
            # Download from URL
            pub_date, raw_text = scrape_document(doc["url"])
            doc["pub_date"] = pub_date or extract_pub_date_from_url(doc["url"])
            if raw_text:
                stats["freshly_downloaded"] += 1
            time.sleep(REQUEST_DELAY)

        if not raw_text:
            doc["text_path"] = None
            stats["failed"] += 1
            continue

        # Clean the text
        clean_text, flags = clean_document_text(
            raw_text, is_pdf, doc_type=doc.get("doc_type"),
        )

        if flags["qa_stripped"]:
            stats["qa_stripped"] += 1
            stats["total_qa_chars_removed"] += flags["qa_chars_removed"]
        if flags["pdf_boilerplate_stripped"]:
            stats["pdf_boilerplate_stripped"] += 1
        if flags["monthly_bulletin_frontmatter_stripped"]:
            stats["monthly_bulletin_frontmatter_stripped"] = (
                stats.get("monthly_bulletin_frontmatter_stripped", 0) + 1
            )
        if flags["looks_like_slide_deck"]:
            stats["suspected_slide_decks"].append({
                "doc_id": doc["raw_id"],
                "doc_type": doc["doc_type"],
                "pub_date": doc.get("pub_date"),
                "first_100": clean_text[:100],
            })

        # Write cleaned file
        if clean_text:
            write_clean_text_file(out_path, doc, clean_text)
        else:
            doc["text_path"] = None
            stats["failed"] += 1
            continue

        # Persist cleaning flags onto doc record
        doc.update(flags)

    return docs, stats


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("ECB DOCUMENT COLLECTION PIPELINE v2")
    log.info(f"Sample: {START_YEAR}–{END_YEAR}")
    log.info(f"Output dir: {RAW_TEXT_DIR.resolve()}")
    log.info("=" * 70)

    # ── Step 1: foedb collection ────────────────────────────────────────────
    log.info("\n── STEP 1: foedb API collection ──")
    version_id, hash_key = get_api_version()
    raw_records = download_all_records(version_id, hash_key)
    foedb_docs = filter_records(raw_records)

    # ── Step 2: deduplication ───────────────────────────────────────────────
    log.info("\n── STEP 2: Deduplicate combined MPS ──")
    foedb_docs = deduplicate_combined_mps(foedb_docs)
    log.info(f"  After dedup: {len(foedb_docs)} foedb documents")

    # ── Step 3: add monthly bulletins ───────────────────────────────────────
    log.info("\n── STEP 3: Add pre-2015 Monthly Bulletins ──")
    monthly_docs = collect_monthly_bulletins()
    docs = foedb_docs + monthly_docs
    log.info(f"  Total documents (foedb + monthly bulletins): {len(docs)}")

    # ── Step 4: process text (clean, save) ──────────────────────────────────
    log.info("\n── STEP 4: Process documents (download/reprocess + clean) ──")
    docs, stats = process_documents(docs)

    log.info("")
    log.info("  Processing summary:")
    log.info(f"    Already clean (skipped):       {stats['already_clean']}")
    log.info(f"    Reprocessed from old raw file: {stats['reprocessed_from_old']}")
    log.info(f"    Freshly downloaded:            {stats['freshly_downloaded']}")
    log.info(f"    Failed:                        {stats['failed']}")
    log.info(f"    Q&A stripped:                  {stats['qa_stripped']} documents, "
             f"{stats['total_qa_chars_removed']:,} chars removed")
    log.info(f"    PDF boilerplate stripped:      {stats['pdf_boilerplate_stripped']} documents")
    log.info(f"    Monthly Bulletin frontmatter:  "
             f"{stats.get('monthly_bulletin_frontmatter_stripped', 0)} documents")

    if stats["suspected_slide_decks"]:
        log.warning("")
        log.warning(f"  ⚠️  {len(stats['suspected_slide_decks'])} documents look like slide decks "
                   "(not in known-deck list).")
        log.warning("  Review these — consider adding their IDs to KNOWN_SLIDE_DECK_IDS:")
        for item in stats["suspected_slide_decks"]:
            log.warning(f"    doc_id={item['doc_id']} type={item['doc_type']} "
                       f"date={item['pub_date']}: {item['first_100']!r}")

    # ── Step 5: save master CSV ─────────────────────────────────────────────
    log.info("\n── STEP 5: Save master CSV ──")
    df = pd.DataFrame(docs)
    # Drop documents that failed processing (no text)
    valid = df[df["text_path"].notna()].copy()
    log.info(f"  Documents with valid text: {len(valid)}/{len(df)}")
    valid.to_csv(MASTER_CSV, index=False)
    log.info(f"  Saved {MASTER_CSV}")

    # ── Final summary by type ───────────────────────────────────────────────
    log.info("")
    log.info("── Final document counts by type ──")
    log.info(f"\n{valid['doc_type'].value_counts().to_string()}")

    log.info("")
    log.info("── By type and year ──")
    pivot = pd.crosstab(valid["year"], valid["doc_type"]).fillna(0).astype(int)
    log.info(f"\n{pivot.to_string()}")

    log.info("\n" + "=" * 70)
    log.info("COLLECTION COMPLETE")
    log.info("=" * 70)
    log.info(f"  Documents:     {len(valid):,}")
    log.info(f"  Text files in: {RAW_TEXT_DIR.resolve()}")
    log.info(f"  Master CSV:    {MASTER_CSV.resolve()}")


if __name__ == "__main__":
    main()
