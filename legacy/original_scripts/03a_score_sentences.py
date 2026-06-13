"""
ECB Sentiment Pipeline v4 — FOMC stance scoring (sentence-level only)
=====================================================================
Augmenting Volatility Models with Text-Based News Analytics
Research Paper — Data Collection & NLP Scoring (v4 final)

DESIGN PRINCIPLE — score-once, iterate-often
--------------------------------------------
The 7-hour bottleneck is the FOMC-RoBERTa forward pass. Everything
downstream of that — aggregation formulas, threshold choices, daily
smoothing — is seconds of pandas. So this script does ONLY the expensive
part:

  1. Read each cleaned ECB document.
  2. NFKC-normalize and split into sentences.
  3. Filter cruft (captions, source lines, table number-soup, doc-type
     front matter) before classification.
  4. Score every surviving sentence with FOMC-RoBERTa
     (gtfintechlab/FOMC-RoBERTa).
  5. Append a row per sentence to sentiment_sentence_level_v4.csv,
     containing FULL sentence text + raw 3-class probabilities + doc
     metadata. No truncation, no document-level aggregation.

All document-level and daily-level aggregation lives in the separate
build_daily_series.py script, which can be re-run cheaply with different
thresholds, smoothing rules, etc.

CHECKPOINTING
-------------
Sentence rows are flushed to disk after each document. If the run crashes
at hour 6, restart picks up from the next unscored doc.

USAGE
-----
Test run (recommended first):
    python ecb_sentiment_pipeline_v4.py --limit 20

Full run (~7 hours on CPU):
    python ecb_sentiment_pipeline_v4.py

Resume an interrupted run (re-reads existing sentence CSV, skips
already-scored doc_ids):
    python ecb_sentiment_pipeline_v4.py --resume

OUTPUT
------
    output/sentiment_sentence_level_v4.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./output")
DOC_MASTER_CSV = OUTPUT_DIR / "ecb_documents_master.csv"

FOMC_MODEL = "gtfintechlab/FOMC-RoBERTa"

# FOMC-RoBERTa label map (verified via huggingface model card)
LABEL_MAP = {
    "LABEL_0": "dovish",
    "LABEL_1": "hawkish",
    "LABEL_2": "neutral",
}

MAX_TOKENS = 512        # RoBERTa max position embeddings
BATCH_SIZE = 16

# Sentence cruft filter (locked in, see discussion in chat)
MIN_SENT_LEN_CHARS = 30          # below this, almost always caption/footer junk
MAX_DIGIT_RATIO    = 0.20        # above this, almost always a table row
CRUFT_REGEX = re.compile(
    r"^\s*("
    # Caption / source / note lines
    r"Latest\s+observations?\s*:|"
    r"Sources?\s*:|"
    r"Notes?\s*:|"
    # Numbered table / chart / box / figure references
    r"Chart\s+\d|Figure\s+\d|Table\s+\d|Box\s+\d|Page\s+\d|Annex\s+[A-Z\d]|"
    # MPS / press conference front matter
    r"PRESS\s+CONFERENCE|"
    r"Introductory\s+statement\s+to\s+the\s+press\s+conference|"
    r"INTRODUCTORY\s+STATEMENT|"
    # Speech front matter / web-export artifacts
    r"Speech\s+by|Keynote\s+(?:speech|address)|Remarks\s+by|"
    r"www\.ecb\.europa\.eu|"
    r"Rubric\s|"
    # Bulletin page header artifacts ("5 ECB Monthly Bulletin Jan 2008")
    r"\d+\s+ECB\s+(?:Monthly|Economic)\s+Bulletin"
    r")",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "pipeline_v4.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT INPUT
# ─────────────────────────────────────────────────────────────────────────────

def read_document_body(text_path):
    """
    Read a cleaned ECB document and return the body text (after the
    metadata header block: PUB_DATE / URL / TYPE / SPEAKER / blank line).

    NOTE: Q&A and PDF boilerplate stripping happen at COLLECTION time
    (ecb_collection_pipeline_v2.py). The body returned here is already
    cleaned text and should be ready to segment + score.
    """
    with open(text_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    parts = re.split(r"\n\s*\n", content, maxsplit=1)
    return parts[1] if len(parts) == 2 else content


def normalize_text(text):
    """
    NFKC unicode normalization: fixes ligatures (ﬁ → fi, ﬂ → fl) and
    composed-character variants. Cheap insurance against PDF extraction
    artifacts.
    """
    return unicodedata.normalize("NFKC", text)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SENTENCE SEGMENTATION + CRUFT FILTER
# ─────────────────────────────────────────────────────────────────────────────

_ABBREVS = ["Mr", "Mrs", "Dr", "Prof", "Fig", "No", "Vol", "pp",
            "i.e", "e.g", "vs", "cf", "et al"]


def split_sentences(text):
    """Regex-based sentence splitter with abbreviation guards."""
    for ab in _ABBREVS:
        text = text.replace(f"{ab}.", f"{ab}DOTPLACEHOLDER")
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.replace("DOTPLACEHOLDER", ".") for s in sentences]
    # Pre-filter on raw length (very wide bounds; the cruft filter is stricter)
    return [s.strip() for s in sentences if 10 <= len(s.strip()) <= 2000]


def clean_sentence(s):
    """Drop URLs and bullet-soup, normalize whitespace."""
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[*•·▪◦\-]{2,}", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_cruft(sentence):
    """
    Return True if this sentence should be DROPPED before classification.
    Checks (in order, cheapest first):
      - too short (< MIN_SENT_LEN_CHARS)
      - too digit-heavy (digit_ratio >= MAX_DIGIT_RATIO)
      - matches the CRUFT_REGEX (caption / source / front matter / etc.)
    """
    if len(sentence) < MIN_SENT_LEN_CHARS:
        return True
    digit_ratio = sum(c.isdigit() for c in sentence) / len(sentence)
    if digit_ratio >= MAX_DIGIT_RATIO:
        return True
    if CRUFT_REGEX.match(sentence):
        return True
    return False


def extract_sentences(body_text):
    """
    Full preprocessing pipeline for one document body:
      1. NFKC normalize.
      2. Segment into sentences.
      3. Clean each sentence.
      4. Drop cruft.
    Returns a list of clean, scoring-ready sentence strings.
    """
    body_text = normalize_text(body_text)
    sentences = split_sentences(body_text)
    sentences = [clean_sentence(s) for s in sentences]
    sentences = [s for s in sentences if not is_cruft(s)]
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# 3. FOMC-ROBERTA SCORING
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers not installed. Run: pip install transformers torch"
        )
    log.info(f"Loading model: {FOMC_MODEL}")
    scorer = pipeline(
        "text-classification",
        model=FOMC_MODEL,
        top_k=None,            # return all 3 class probabilities
        truncation=True,
        max_length=MAX_TOKENS,
        device=-1,             # CPU
    )
    log.info("Model loaded.")
    return scorer


def score_sentences_batch(scorer, sentences, batch_size=BATCH_SIZE):
    """
    Run sentences through the classifier in batches. Returns a list of
    dicts: {sentence, dovish_score, hawkish_score, neutral_score}.
    On per-batch failure, rows are filled with uniform 1/3 probabilities
    (an unmistakable diagnostic flag).
    """
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            outputs = scorer(batch)
            for sent, out in zip(batch, outputs):
                scores = {LABEL_MAP[item["label"]]: item["score"] for item in out}
                results.append({
                    "sentence":      sent,
                    "dovish_score":  scores.get("dovish",  0.0),
                    "hawkish_score": scores.get("hawkish", 0.0),
                    "neutral_score": scores.get("neutral", 0.0),
                })
        except Exception as e:
            log.warning(f"  Batch starting at sentence-index {i} failed: {e}")
            for sent in batch:
                results.append({
                    "sentence": sent,
                    "dovish_score": 1/3, "hawkish_score": 1/3, "neutral_score": 1/3,
                })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHECKPOINTING (per-document append)
# ─────────────────────────────────────────────────────────────────────────────

SENTENCE_CSV_COLUMNS = [
    "doc_id", "pub_date", "doc_type", "speaker",
    "sentence_idx", "sentence",
    "dovish_score", "hawkish_score", "neutral_score",
]


def append_sentence_rows(out_csv, rows):
    """Append rows to the sentence-level CSV. Creates header on first write."""
    df = pd.DataFrame(rows, columns=SENTENCE_CSV_COLUMNS)
    write_header = not out_csv.exists()
    df.to_csv(out_csv, mode="a", header=write_header, index=False)


def already_scored_doc_ids(out_csv):
    """
    Return the set of doc_ids that already appear in the sentence-level
    CSV. Used by --resume to skip already-completed documents.
    """
    if not out_csv.exists():
        return set()
    try:
        existing = pd.read_csv(out_csv, usecols=["doc_id"])
        return set(existing["doc_id"].astype(str).unique())
    except Exception as e:
        log.warning(f"  Could not parse existing sentence CSV ({e}); starting fresh.")
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Score only the first N documents (mixed across types) for testing.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip docs already present in the sentence CSV and continue.",
    )
    parser.add_argument(
        "--doc-types", default=None,
        help="Comma-separated doc_types to include "
             "(e.g. 'monthly_bulletin' to score only monthly bulletins). "
             "Default: all doc_types.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for the sentence-level CSV output. Defaults: "
             "sentiment_sentence_level_v4.csv (full run); "
             "sentiment_sentence_level_v4_test{N}.csv (--limit); "
             "sentiment_sentence_level_v4_freshly_scored.csv (--doc-types).",
    )
    args = parser.parse_args()

    # ── Output paths ────────────────────────────────────────────────────────
    if args.output:
        sent_csv = Path(args.output)
        if not sent_csv.is_absolute():
            sent_csv = OUTPUT_DIR / sent_csv.name
    elif args.limit:
        sent_csv = OUTPUT_DIR / f"sentiment_sentence_level_v4_test{args.limit}.csv"
    elif args.doc_types:
        # Partial run (e.g. just monthly bulletins) → goes to the file the
        # merge script expects to combine with old-source rows.
        sent_csv = OUTPUT_DIR / "sentiment_sentence_level_v4_freshly_scored.csv"
    else:
        sent_csv = OUTPUT_DIR / "sentiment_sentence_level_v4.csv"

    log.info("=" * 70)
    log.info("ECB SENTIMENT PIPELINE v4 — sentence-level scoring only")
    log.info(f"  Cruft filter:  len>={MIN_SENT_LEN_CHARS}, "
             f"digit_ratio<{MAX_DIGIT_RATIO}, regex patterns")
    log.info(f"  Output:        {sent_csv}")
    if args.limit:
        log.info(f"  TEST RUN:      limit={args.limit}")
    if args.resume:
        log.info(f"  RESUME:        ON")
    log.info("=" * 70)

    # ── Determine which docs to score ───────────────────────────────────────
    if not DOC_MASTER_CSV.exists():
        raise FileNotFoundError(f"{DOC_MASTER_CSV} not found.")
    docs = pd.read_csv(DOC_MASTER_CSV)
    log.info(f"  Master CSV: {len(docs):,} documents")

    if args.doc_types:
        wanted = [s.strip() for s in args.doc_types.split(",") if s.strip()]
        before = len(docs)
        docs = docs[docs["doc_type"].isin(wanted)].reset_index(drop=True)
        log.info(f"  Filtered to doc_types {wanted}: {len(docs)} of {before}")

    if args.limit:
        # For test runs, take a balanced mix across doc_types.
        type_samples = []
        per_type = max(1, args.limit // docs["doc_type"].nunique())
        for dt in docs["doc_type"].dropna().unique():
            type_samples.append(docs[docs["doc_type"] == dt].head(per_type))
        docs = pd.concat(type_samples).head(args.limit).reset_index(drop=True)
        log.info(f"  Limited to {len(docs)} docs: {dict(docs['doc_type'].value_counts())}")

    if args.resume:
        scored = already_scored_doc_ids(sent_csv)
        before = len(docs)
        docs = docs[~docs["raw_id"].astype(str).isin(scored)].reset_index(drop=True)
        log.info(f"  Resume: skipping {before - len(docs)} already-scored docs; "
                 f"{len(docs)} remain.")

    if len(docs) == 0:
        log.info("Nothing to score. Exiting.")
        return

    # ── Load model ──────────────────────────────────────────────────────────
    scorer = load_model()

    # ── Score loop ──────────────────────────────────────────────────────────
    n_docs_scored = 0
    n_sentences_total = 0
    n_sentences_filtered_total = 0

    for _, doc in tqdm(docs.iterrows(), total=len(docs), desc="Scoring docs"):
        text_path = doc.get("text_path")
        if not isinstance(text_path, str) or not os.path.exists(text_path):
            continue

        body = read_document_body(text_path)
        # Count pre-filter sentence count for diagnostics
        raw_sents = split_sentences(normalize_text(body))
        sentences = extract_sentences(body)
        n_filtered = len(raw_sents) - len(sentences)
        n_sentences_filtered_total += n_filtered

        if not sentences:
            continue

        scored = score_sentences_batch(scorer, sentences)

        rows = [
            {
                "doc_id":        doc["raw_id"],
                "pub_date":      doc.get("pub_date"),
                "doc_type":      doc.get("doc_type"),
                "speaker":       doc.get("speaker"),
                "sentence_idx":  idx,
                "sentence":      s["sentence"],   # FULL text, no truncation
                "dovish_score":  round(s["dovish_score"],  6),
                "hawkish_score": round(s["hawkish_score"], 6),
                "neutral_score": round(s["neutral_score"], 6),
            }
            for idx, s in enumerate(scored)
        ]
        append_sentence_rows(sent_csv, rows)

        n_docs_scored += 1
        n_sentences_total += len(scored)

    log.info("")
    log.info(f"  Documents scored:        {n_docs_scored:,}")
    log.info(f"  Sentences scored:        {n_sentences_total:,}")
    log.info(f"  Sentences filtered out:  {n_sentences_filtered_total:,} "
             f"(cruft / length / digit-ratio)")
    log.info(f"  Output: {sent_csv}")
    log.info("=" * 70)
    log.info("Done. Next: run build_daily_series.py to produce daily series.")


if __name__ == "__main__":
    main()
