"""
build_daily_series.py — post-classification aggregation for v4 pipeline
=======================================================================

Consumes:
    output/sentiment_sentence_level_v4.csv

Produces:
    output/sentiment_document_level_v4.csv         (one row per doc)
    output/sentiment_daily_all_sources_v4.csv      (one row per calendar day)
    output/sentiment_daily_mps_only_v4.csv
    output/sentiment_daily_speeches_only_v4.csv

DESIGN
------
This is the "fast iterate" half of the score-once, iterate-often
pipeline. It is computed deterministically from the saved sentence-level
scores, so we can re-run with different parameters in seconds.

DOCUMENT-LEVEL AGGREGATION (per spec, NA-GARCH-compatible)
----------------------------------------------------------
For each document and for each confidence threshold τ ∈ {0.00, 0.50, 0.80}:

    is_dovish_τ(s)   = (argmax class is dovish)  AND (max_prob ≥ τ)
    is_hawkish_τ(s)  = (argmax class is hawkish) AND (max_prob ≥ τ)

    n_total          = total sentences (after pre-classification cruft filter)
    n_dovish_τ       = #sentences where is_dovish_τ
    n_hawkish_τ      = #sentences where is_hawkish_τ
    sum_dovish_τ     = Σ dovish_prob over sentences where is_dovish_τ
    sum_hawkish_τ    = Σ hawkish_prob over sentences where is_hawkish_τ

    P_doc_dovish_τ   = sum_dovish_τ  / n_total      ∈ [0, 1]
    P_doc_hawkish_τ  = sum_hawkish_τ / n_total      ∈ [0, 1]
    policy_intensity_τ = (n_dovish_τ + n_hawkish_τ) / n_total

The TOTAL-sentences denominator (rather than non-neutral) avoids the
structural collinearity issue and keeps the directional series
interpretable; topic-relevance is captured separately via
`policy_intensity`.

DAILY AGGREGATION
-----------------
Multi-doc days: equal-weight average across all documents published on
that calendar day (so a long bulletin doesn't mechanically drown out a
short speech). A pooled-sentences alternative is also computed for
robustness.

SADIK-STYLE RESCALING (Eq 3.2 of Sadik, Date & Mitra 2018)
----------------------------------------------------------
After daily aggregation, each series is rescaled by its IN-SAMPLE
maximum so that the in-sample range is [0, 1] for dovish and [-1, 0]
for the negated-hawkish term. Out-of-sample values can exceed those
bounds (this is honest — out-of-sample magnitudes are not known at
estimation time).

SMOOTHING
---------
Persist-to-next-event (the spec): each event-day's stance value
forward-fills until the next event-day. A no-smoothing event-only
series is also retained for diagnostic use.

USAGE
-----
    python build_daily_series.py
    python build_daily_series.py --sentence-csv output/sentiment_sentence_level_v4_test20.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./output")

THRESHOLDS = [0.00, 0.50, 0.80]
THRESHOLD_LABELS = {0.00: "t000", 0.50: "t050", 0.80: "t080"}

# In-sample window for Sadik-style rescaling (matches paper plan)
IN_SAMPLE_START = pd.Timestamp("2008-01-01")
IN_SAMPLE_END   = pd.Timestamp("2018-12-31")

# Daily series calendar coverage
SERIES_START = pd.Timestamp("2008-01-01")
SERIES_END   = pd.Timestamp("2023-12-31")

# Doc-type groupings for source-specific daily series
MPS_TYPES = [
    "monetary_policy_statement",
    "combined_monetary_policy_statement",
    "governing_council_statement",
]
SPEECH_TYPES = ["speech"]
BULLETIN_TYPES = ["monthly_bulletin", "economic_bulletin"]


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "build_daily_series.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOCUMENT-LEVEL AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def annotate_argmax(sent_df):
    """
    Add max_class and max_prob columns to the sentence-level DataFrame.
    """
    probs = sent_df[["dovish_score", "hawkish_score", "neutral_score"]].to_numpy()
    sent_df = sent_df.copy()
    argmax_idx = probs.argmax(axis=1)
    classes = np.array(["dovish", "hawkish", "neutral"])
    sent_df["max_class"] = classes[argmax_idx]
    sent_df["max_prob"]  = probs.max(axis=1)
    return sent_df


def aggregate_documents(sent_df):
    """
    Group sentences by document and produce one row per document.
    Computes counts, probability sums, and ratios at every threshold.
    """
    log.info("  Annotating argmax / max_prob ...")
    sent_df = annotate_argmax(sent_df)

    log.info(f"  Aggregating {sent_df['doc_id'].nunique():,} documents "
             f"from {len(sent_df):,} sentences ...")

    out_rows = []
    for doc_id, g in sent_df.groupby("doc_id", sort=False):
        n_total = len(g)
        row = {
            "doc_id":   doc_id,
            "pub_date": g["pub_date"].iloc[0],
            "doc_type": g["doc_type"].iloc[0],
            "speaker":  g["speaker"].iloc[0],
            "n_total":  n_total,
        }
        for tau in THRESHOLDS:
            label = THRESHOLD_LABELS[tau]
            mask = g["max_prob"] >= tau
            d_mask = mask & (g["max_class"] == "dovish")
            h_mask = mask & (g["max_class"] == "hawkish")
            n_d = int(d_mask.sum())
            n_h = int(h_mask.sum())
            sum_d = float(g.loc[d_mask, "dovish_score"].sum())
            sum_h = float(g.loc[h_mask, "hawkish_score"].sum())
            row[f"n_dovish_{label}"]       = n_d
            row[f"n_hawkish_{label}"]      = n_h
            row[f"sum_dovish_{label}"]     = round(sum_d, 6)
            row[f"sum_hawkish_{label}"]    = round(sum_h, 6)
            row[f"P_doc_dovish_{label}"]   = round(sum_d / n_total, 6)
            row[f"P_doc_hawkish_{label}"]  = round(sum_h / n_total, 6)
            row[f"policy_intensity_{label}"] = round((n_d + n_h) / n_total, 6)
        out_rows.append(row)

    doc_df = pd.DataFrame(out_rows)
    doc_df["pub_date"] = pd.to_datetime(doc_df["pub_date"], errors="coerce")
    doc_df = doc_df.sort_values(["pub_date", "doc_id"]).reset_index(drop=True)
    return doc_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DAILY AGGREGATION (equal-weight per document)
# ─────────────────────────────────────────────────────────────────────────────

def daily_event_series(doc_df, source_filter=None):
    """
    Produce one row per CALENDAR DAY in [SERIES_START, SERIES_END].

    For event days (≥1 doc published), values are the equal-weight mean
    across documents of that day. For non-event days, values are NaN
    (event-only series; smoothing is applied afterwards).
    """
    df = doc_df.copy()
    if source_filter is not None:
        df = df[df["doc_type"].isin(source_filter)]

    df = df.dropna(subset=["pub_date"])
    df = df[(df["pub_date"] >= SERIES_START) & (df["pub_date"] <= SERIES_END)]
    df = df.sort_values("pub_date")

    # Columns to mean-aggregate per day
    agg_cols = []
    for tau in THRESHOLDS:
        label = THRESHOLD_LABELS[tau]
        agg_cols += [
            f"P_doc_dovish_{label}",
            f"P_doc_hawkish_{label}",
            f"policy_intensity_{label}",
        ]

    # Equal-weight mean per day
    by_day = df.groupby(df["pub_date"].dt.normalize())[agg_cols].mean()
    n_docs_per_day = df.groupby(df["pub_date"].dt.normalize()).size().rename("n_docs_today")

    # Calendar-day reindex
    all_days = pd.date_range(SERIES_START, SERIES_END, freq="D")
    daily = by_day.reindex(all_days)
    daily.index.name = "date"
    daily["is_event_day"] = daily[agg_cols[0]].notna().astype(int)
    daily = daily.join(n_docs_per_day.reindex(all_days).fillna(0).astype(int))

    return daily.reset_index()


def apply_persist_smoothing(daily_df):
    """
    Forward-fill (persist-to-next-event) the directional / intensity
    columns. Adds parallel columns with `_persist` suffix; the
    event-only originals are kept.

    Days before the first event remain NaN in both series.
    """
    out = daily_df.copy()
    for tau in THRESHOLDS:
        label = THRESHOLD_LABELS[tau]
        for stem in ("P_doc_dovish", "P_doc_hawkish", "policy_intensity"):
            col = f"{stem}_{label}"
            out[f"{col}_persist"] = out[col].ffill()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. SADIK-STYLE IN-SAMPLE MAX RESCALING
# ─────────────────────────────────────────────────────────────────────────────

def apply_sadik_rescaling(daily_df):
    """
    Rescale persisted directional series by the in-sample maximum so
    that in-sample P_t ∈ [0, 1] and N_t ∈ [-1, 0] (Sadik et al. 2018,
    Eq 3.2). Out-of-sample values may exceed these bounds — this is
    intentional (the in-sample max is fixed at estimation time).

    Adds:
      P_t_<thr>     = P_doc_dovish_<thr>_persist  / max_in_sample
      N_t_<thr>     = -P_doc_hawkish_<thr>_persist / max_in_sample
      S_t_<thr>     = P_t_<thr> + N_t_<thr>          (net stance signal)
    """
    out = daily_df.copy()
    in_sample = (out["date"] >= IN_SAMPLE_START) & (out["date"] <= IN_SAMPLE_END)

    rescale_info = {}
    for tau in THRESHOLDS:
        label = THRESHOLD_LABELS[tau]
        d_col = f"P_doc_dovish_{label}_persist"
        h_col = f"P_doc_hawkish_{label}_persist"

        d_max = out.loc[in_sample, d_col].max(skipna=True)
        h_max = out.loc[in_sample, h_col].max(skipna=True)
        rescale_info[label] = {"d_max_is": d_max, "h_max_is": h_max}

        # Guard against degenerate zero-max
        d_div = d_max if (d_max and d_max > 0) else 1.0
        h_div = h_max if (h_max and h_max > 0) else 1.0

        out[f"P_t_{label}"] = out[d_col] / d_div
        out[f"N_t_{label}"] = -(out[h_col] / h_div)
        out[f"S_t_{label}"] = out[f"P_t_{label}"] + out[f"N_t_{label}"]

    return out, rescale_info


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVENT-STUDY SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

EVENTS_OF_INTEREST = [
    ("Draghi 'whatever it takes'",
     "2012-07-26", "expected dovish (+P shift, market-calming)"),
    ("Draghi 'OMT' announcement",
     "2012-09-06", "expected dovish"),
    ("COVID emergency PEPP",
     "2020-03-18", "expected dovish (emergency easing)"),
    ("First post-COVID hike",
     "2022-07-21", "expected hawkish"),
    ("75bp hike (anti-inflation)",
     "2022-09-08", "expected hawkish"),
    ("Last hike of cycle",
     "2023-09-14", "expected hawkish (or peak hawkish)"),
]


def event_study_printout(doc_df):
    """
    For each named event, print the document-level scores around the
    event date (within ±7 calendar days), so we can sanity-check whether
    the directional reading matches the intuitive label.
    """
    log.info("\n  Event-study spot checks (±7 days around named events):")
    for name, date_s, expectation in EVENTS_OF_INTEREST:
        d = pd.to_datetime(date_s)
        window_lo = d - pd.Timedelta(days=7)
        window_hi = d + pd.Timedelta(days=7)
        sub = doc_df[
            (doc_df["pub_date"] >= window_lo)
            & (doc_df["pub_date"] <= window_hi)
        ].copy()
        log.info(f"\n    [{name}]  {date_s}  — {expectation}")
        if sub.empty:
            log.info(f"      (no documents in window)")
            continue
        # Print spec-aligned (t050) directional scores per doc
        for _, r in sub.iterrows():
            log.info(
                f"      {r['pub_date'].date()}  {r['doc_type']:<35} "
                f"P_dov={r.get('P_doc_dovish_t050', float('nan')):.3f} "
                f"P_haw={r.get('P_doc_hawkish_t050', float('nan')):.3f} "
                f"intens={r.get('policy_intensity_t050', float('nan')):.3f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def write_daily(daily_df, out_path, label):
    daily_df.to_csv(out_path, index=False)
    n_event = int(daily_df["is_event_day"].sum())
    log.info(f"    {label:<22} {len(daily_df):,} days, "
             f"{n_event:,} event-days  →  {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentence-csv",
        default=str(OUTPUT_DIR / "sentiment_sentence_level_v4.csv"),
        help="Sentence-level CSV produced by ecb_sentiment_pipeline_v4.py",
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help="Optional suffix for output files (e.g. '_test20')",
    )
    args = parser.parse_args()

    sent_csv = Path(args.sentence_csv)
    suffix   = args.out_suffix
    if not sent_csv.exists():
        raise FileNotFoundError(f"{sent_csv} not found")

    log.info("=" * 70)
    log.info("BUILD DAILY SERIES — v4")
    log.info(f"  Input:  {sent_csv}")
    log.info(f"  Suffix: '{suffix}'" if suffix else "  Suffix: (none)")
    log.info("=" * 70)

    # ── 1. Document-level aggregation ───────────────────────────────────────
    log.info("\n── Step 1: Read sentence-level scores ──")
    sent_df = pd.read_csv(sent_csv)
    log.info(f"  Sentences: {len(sent_df):,} across {sent_df['doc_id'].nunique():,} docs")

    log.info("\n── Step 2: Document-level aggregation ──")
    doc_df = aggregate_documents(sent_df)

    doc_csv = OUTPUT_DIR / f"sentiment_document_level_v4{suffix}.csv"
    doc_df.to_csv(doc_csv, index=False)
    log.info(f"  Saved → {doc_csv}")

    # Quick doc-level diagnostic
    log.info("\n  Doc-level summary at threshold 0.50:")
    log.info(f"    P_doc_dovish_t050:  mean={doc_df['P_doc_dovish_t050'].mean():.3f}, "
             f"std={doc_df['P_doc_dovish_t050'].std():.3f}, "
             f"max={doc_df['P_doc_dovish_t050'].max():.3f}")
    log.info(f"    P_doc_hawkish_t050: mean={doc_df['P_doc_hawkish_t050'].mean():.3f}, "
             f"std={doc_df['P_doc_hawkish_t050'].std():.3f}, "
             f"max={doc_df['P_doc_hawkish_t050'].max():.3f}")
    log.info(f"    policy_intensity_t050: mean={doc_df['policy_intensity_t050'].mean():.3f}, "
             f"std={doc_df['policy_intensity_t050'].std():.3f}")
    corr = doc_df["P_doc_dovish_t050"].corr(doc_df["P_doc_hawkish_t050"])
    log.info(f"    corr(P_dovish, P_hawkish)_t050 = {corr:+.3f}")

    # ── 2. Event-study sanity check ─────────────────────────────────────────
    log.info("\n── Step 3: Event-study spot checks ──")
    event_study_printout(doc_df)

    # ── 3. Daily aggregation + smoothing + Sadik rescaling ──────────────────
    log.info("\n── Step 4: Daily aggregation, smoothing, Sadik rescaling ──")

    runs = [
        ("all_sources",   None),
        ("mps_only",      MPS_TYPES),
        ("speeches_only", SPEECH_TYPES),
        ("bulletins_only", BULLETIN_TYPES),
    ]
    for stem, src in runs:
        daily = daily_event_series(doc_df, source_filter=src)
        daily = apply_persist_smoothing(daily)
        daily, rescale_info = apply_sadik_rescaling(daily)
        out_path = OUTPUT_DIR / f"sentiment_daily_{stem}_v4{suffix}.csv"
        write_daily(daily, out_path, stem)
        # Log the in-sample max used for rescaling (for the t050 series)
        info = rescale_info["t050"]
        log.info(f"      Sadik in-sample max @t050: dovish={info['d_max_is']:.4f}, "
                 f"hawkish={info['h_max_is']:.4f}")

    log.info("\n" + "=" * 70)
    log.info("BUILD DAILY SERIES COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
