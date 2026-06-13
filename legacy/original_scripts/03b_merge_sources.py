"""
merge_v4_sources.py — assemble the canonical v4 sentence-level CSV
==================================================================

Combines two scoring sources into a single
`output/sentiment_sentence_level_v4.csv` ready for `build_daily_series.py`:

    1. Old `sentiment_sentence_level_fomc.csv` — already-scored 1,099 docs
       from the previous (v3) pipeline run with FOMC-RoBERTa. Used for
       speeches, MPS, and economic bulletins (1,079 of these are in the
       current corpus). One MPS doc is dropped explicitly (see below).

    2. Fresh `sentiment_sentence_level_v4.csv` — the 84 newly-added
       monthly bulletins (2008-2014), scored by the v4 pipeline with
       NFKC normalization and the cruft filter.

PROCESSING APPLIED TO THE OLD CSV
---------------------------------
    a. Restrict to doc_ids present in the current master CSV.
    b. Drop doc_id 10260 (2014 "comprehensive assessment" press conference;
       not a standard monetary policy statement, format is hybrid).
    c. Q&A boundary filter: for each doc, find the earliest sentence
       matching either
         (i)  the canonical ECB transition phrase, OR
         (ii) the case-sensitive `Question:` fallback,
       and DROP that sentence and everything after.
    d. Apply v4 sentence-level cruft filter (length / digit-ratio /
       cruft-regex) post-hoc.

Steps (a)–(d) are applied uniformly to all doc_types from the old CSV.
False-positive matches of the boundary phrase in non-MPS docs amount to
~50 closing speech lines (~0.03% of sentences) — verified harmless.

OUTPUT
------
    output/sentiment_sentence_level_v4.csv     ← canonical input for build_daily_series.py

USAGE
-----
    python merge_v4_sources.py
"""

from __future__ import annotations

import logging
import re
import sys
import types
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./output")
OLD_FOMC_CSV    = OUTPUT_DIR / "sentiment_sentence_level_fomc.csv"
FRESH_V4_CSV    = OUTPUT_DIR / "sentiment_sentence_level_v4_freshly_scored.csv"
FINAL_V4_CSV    = OUTPUT_DIR / "sentiment_sentence_level_v4.csv"
DOC_MASTER_CSV  = OUTPUT_DIR / "ecb_documents_master.csv"

# doc_id of the 2014 "comprehensive assessment press conference" — drop entirely
# (non-standard MPS, hybrid intro/Q&A format, no clean boundary).
DROP_DOC_IDS = {"10260"}

# ─── Q&A boundary patterns ──────────────────────────────────────────────────
# Match the v2 collection-pipeline patch exactly so behavior is consistent.
QA_BOUNDARY_CI = re.compile(
    r"(?:We|I)\s+(?:are|am)\s+now\s+(?:at\s+your\s+disposal|ready\s+to\s+take|"
    r"happy\s+to\s+take|here\s+to\s+take|happy|ready|glad)\b"
    r"|"
    r"Transcript of the questions asked",
    re.IGNORECASE,
)
QA_BOUNDARY_CS = re.compile(r"(?:^|[\s.,])Question\s*[:.]\s+[A-Z]")


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "merge_v4_sources.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHARED CRUFT FILTER (imports v4's filter to stay in lockstep)
# ─────────────────────────────────────────────────────────────────────────────

def _import_v4_cruft_filter():
    """
    Import is_cruft directly from ecb_sentiment_pipeline_v4 so the merge script
    and the live scoring use IDENTICAL filter semantics. transformers is stubbed
    because we don't need the model here.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    sys.modules.setdefault(
        "transformers", types.SimpleNamespace(pipeline=None)
    )
    from ecb_sentiment_pipeline_v4 import is_cruft  # noqa: E402
    return is_cruft


is_cruft = _import_v4_cruft_filter()


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-DOC Q&A BOUNDARY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_qa_boundary_idx(group_df):
    """
    For sentences of one document (sorted by sentence_idx), return the
    earliest sentence_idx whose text matches a Q&A boundary marker, or
    None if no marker found.
    """
    for _, r in group_df.iterrows():
        s = r["sentence"]
        if not isinstance(s, str):
            continue
        if QA_BOUNDARY_CI.search(s) or QA_BOUNDARY_CS.search(s):
            return r["sentence_idx"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROCESS THE OLD FOMC CSV
# ─────────────────────────────────────────────────────────────────────────────

def process_old_csv(current_doc_ids):
    """
    Returns a DataFrame in v4 sentence-CSV schema (doc_id, pub_date,
    doc_type, speaker, sentence_idx, sentence, dovish_score, hawkish_score,
    neutral_score), with:
      - rows restricted to current corpus
      - dropped DROP_DOC_IDS
      - per-doc Q&A boundary applied (everything from boundary onward removed)
      - cruft filter applied
    """
    log.info("\n── Reading old fomc CSV ──")
    df = pd.read_csv(OLD_FOMC_CSV)
    log.info(f"  Old CSV rows: {len(df):,} across {df['doc_id'].nunique():,} docs")

    # (a) restrict to current corpus
    df["doc_id_str"] = df["doc_id"].astype(str)
    df = df[df["doc_id_str"].isin(current_doc_ids)].copy()
    log.info(f"  After restricting to current corpus: "
             f"{len(df):,} rows / {df['doc_id'].nunique():,} docs")

    # (b) drop explicit outlier docs
    n_before = len(df)
    df = df[~df["doc_id_str"].isin(DROP_DOC_IDS)].copy()
    log.info(f"  Dropped explicit outliers ({sorted(DROP_DOC_IDS)}): "
             f"{n_before - len(df):,} rows removed")

    # (c) Q&A boundary filter, per document
    log.info(f"  Detecting Q&A boundaries per doc ...")
    df = df.sort_values(["doc_id", "sentence_idx"]).reset_index(drop=True)
    boundaries = (
        df.groupby("doc_id_str", sort=False)
          .apply(find_qa_boundary_idx, include_groups=False)
          .rename("qa_boundary_idx")
          .reset_index()
    )
    df = df.merge(boundaries, on="doc_id_str", how="left")
    n_with_boundary = boundaries["qa_boundary_idx"].notna().sum()
    log.info(f"    {n_with_boundary} of {len(boundaries)} docs had a detectable Q&A boundary")

    n_before = len(df)
    keep_mask = df["qa_boundary_idx"].isna() | (df["sentence_idx"] < df["qa_boundary_idx"])
    df = df[keep_mask].copy()
    log.info(f"    Q&A filter dropped {n_before - len(df):,} sentences "
             f"({(n_before - len(df))/n_before:.1%})")

    # (d) cruft filter
    log.info(f"  Applying v4 cruft filter ...")
    df["v4_keep"] = df["sentence"].apply(
        lambda s: not is_cruft(s) if isinstance(s, str) else False
    )
    n_before = len(df)
    df = df[df["v4_keep"]].copy()
    log.info(f"    Cruft filter dropped {n_before - len(df):,} sentences "
             f"({(n_before - len(df))/n_before:.1%})")

    # Drop helper columns and conform schema
    df = df.drop(columns=["doc_id_str", "qa_boundary_idx", "v4_keep"])
    df = df[[
        "doc_id", "pub_date", "doc_type", "speaker",
        "sentence_idx", "sentence",
        "dovish_score", "hawkish_score", "neutral_score",
    ]]

    log.info(f"  Final from old CSV: {len(df):,} rows / {df['doc_id'].nunique():,} docs")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. ATTACH THE FRESHLY-SCORED MONTHLY BULLETINS
# ─────────────────────────────────────────────────────────────────────────────

def load_fresh_csv():
    if not FRESH_V4_CSV.exists():
        log.warning(
            f"  Fresh v4 CSV not found at {FRESH_V4_CSV}. "
            f"Skipping fresh-source merge — final CSV will contain old-source rows only. "
            f"To produce it, run "
            f"`python ecb_sentiment_pipeline_v4.py --doc-types monthly_bulletin` "
            f"(it writes to {FRESH_V4_CSV.name} automatically) and re-run this script."
        )
        return None
    log.info(f"\n── Reading freshly-scored v4 CSV ──")
    df = pd.read_csv(FRESH_V4_CSV)
    log.info(f"  Fresh CSV rows: {len(df):,} across {df['doc_id'].nunique():,} docs")
    by_type = df["doc_type"].value_counts().to_dict()
    log.info(f"  By doc_type: {by_type}")

    # The fresh v4 CSV is written with v4's segmentation + cruft filter applied,
    # so no further filtering is needed here.
    return df[[
        "doc_id", "pub_date", "doc_type", "speaker",
        "sentence_idx", "sentence",
        "dovish_score", "hawkish_score", "neutral_score",
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("MERGE V4 SOURCES — assemble canonical sentiment_sentence_level_v4.csv")
    log.info("=" * 70)

    if not DOC_MASTER_CSV.exists():
        raise FileNotFoundError(f"{DOC_MASTER_CSV} not found")
    if not OLD_FOMC_CSV.exists():
        raise FileNotFoundError(f"{OLD_FOMC_CSV} not found")

    master = pd.read_csv(DOC_MASTER_CSV)
    current_doc_ids = set(master["raw_id"].astype(str).unique())
    log.info(f"\n  Current master corpus: {len(current_doc_ids):,} docs")

    # 1. Old CSV processing
    old_df = process_old_csv(current_doc_ids)

    # 2. Fresh CSV (monthly bulletins)
    fresh_df = load_fresh_csv()

    # 3. Concatenate, dedupe (defensive: a doc appearing in both sources
    #    keeps the FRESH version; this should never happen with current
    #    doc_id partitioning but cheap insurance.)
    if fresh_df is not None:
        old_only = old_df[~old_df["doc_id"].astype(str).isin(
            fresh_df["doc_id"].astype(str)
        )]
        if len(old_only) < len(old_df):
            log.warning(
                f"  {len(old_df) - len(old_only)} rows in old CSV had doc_ids "
                f"also present in fresh CSV — preferring fresh."
            )
        merged = pd.concat([old_only, fresh_df], ignore_index=True)
    else:
        merged = old_df

    # 4. Sort by date / doc_id / sentence_idx for deterministic output
    merged = merged.sort_values(
        ["pub_date", "doc_id", "sentence_idx"],
        kind="mergesort",
    ).reset_index(drop=True)

    # ── Write final CSV ─────────────────────────────────────────────────────
    log.info(f"\n── Writing canonical v4 sentence-level CSV ──")
    merged.to_csv(FINAL_V4_CSV, index=False)
    log.info(f"  → {FINAL_V4_CSV}")
    log.info(f"  Rows: {len(merged):,}")
    log.info(f"  Docs: {merged['doc_id'].nunique():,}")
    log.info(f"  By doc_type: {merged['doc_type'].value_counts().to_dict()}")

    # Quick sanity check: doc-type retention against master
    master_by_type = master["doc_type"].value_counts().to_dict()
    merged_by_type = merged.groupby("doc_id").first()["doc_type"].value_counts().to_dict()
    log.info("\n  Coverage vs. master CSV:")
    for t in sorted(set(master_by_type) | set(merged_by_type)):
        a = master_by_type.get(t, 0)
        b = merged_by_type.get(t, 0)
        log.info(f"    {t:<35} master={a:<5} scored={b:<5}  "
                 f"({b/a:.0%} coverage)" if a else f"    {t}: 0/0")

    log.info("\n" + "=" * 70)
    log.info("MERGE COMPLETE — next: python build_daily_series.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
