"""
diagnostics_v4.py — pre-modeling visual sanity checks
=====================================================

Produces a set of PNG plots in output/diagnostics_v4/ plus a text summary.
Run AFTER build_daily_series.py.

What it covers:
    1. Daily time series with crisis annotations
    2. Doc-level distributions of P_doc / N_doc / intensity by doc_type
    3. Scatter of P_doc vs P_hawkish (correlation structure)
    4. Annual means of P, N, intensity (structural-break check)
    5. Sentence-level class distribution by doc_type
    6. Threshold sensitivity (t000 / t050 / t080) on P_t and N_t

USAGE
-----
    python diagnostics_v4.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./output")
DIAG_DIR   = OUTPUT_DIR / "diagnostics_v4"

SENT_CSV   = OUTPUT_DIR / "sentiment_sentence_level_v4.csv"
DOC_CSV    = OUTPUT_DIR / "sentiment_document_level_v4.csv"
DAILY_ALL  = OUTPUT_DIR / "sentiment_daily_all_sources_v4.csv"

# Crisis / event annotations to mark on time-series
CRISIS_EVENTS = [
    ("Lehman", "2008-09-15"),
    ("ECB rate cut", "2008-10-08"),
    ("Sov. debt crisis", "2010-05-10"),
    ("Draghi 'whatever it takes'", "2012-07-26"),
    ("OMT", "2012-09-06"),
    ("APP launch", "2015-01-22"),
    ("COVID PEPP", "2020-03-18"),
    ("Russia/Ukraine", "2022-02-24"),
    ("First hike (post-COVID)", "2022-07-21"),
    ("75bp hike", "2022-09-08"),
]

# Style
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def annotate_events(ax, ymin=None, ymax=None, fontsize=7):
    if ymin is None:
        ymin, ymax = ax.get_ylim()
    for label, date in CRISIS_EVENTS:
        d = pd.to_datetime(date)
        ax.axvline(d, color="grey", lw=0.6, alpha=0.5)
        ax.text(d, ymax, " " + label, rotation=90, va="top", ha="left",
                fontsize=fontsize, color="grey")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DAILY SERIES TIME-SERIES
# ─────────────────────────────────────────────────────────────────────────────

def plot_daily_series(daily, out):
    """Three-panel time series: P_t, N_t, intensity. Persist-smoothed (Sadik-rescaled)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    d = daily.set_index("date")

    # Panel 1: P_t (rescaled, persist)
    axes[0].plot(d.index, d["P_t_t050"], lw=0.8, color="tab:blue", label="P_t (dovish, Sadik-rescaled, persist)")
    axes[0].set_ylabel("P_t (dovish)")
    axes[0].set_title("Daily dovish stance signal — Sadik-rescaled, persist-to-next-event")
    axes[0].axhline(0, color="black", lw=0.4)
    annotate_events(axes[0])

    # Panel 2: N_t (rescaled, persist) — already negative
    axes[1].plot(d.index, d["N_t_t050"], lw=0.8, color="tab:red", label="N_t (hawkish, negated, persist)")
    axes[1].set_ylabel("N_t (hawkish, negated)")
    axes[1].set_title("Daily hawkish stance signal (negated, per Sadik convention)")
    axes[1].axhline(0, color="black", lw=0.4)
    annotate_events(axes[1])

    # Panel 3: intensity (persist) raw, not rescaled
    axes[2].plot(d.index, d["policy_intensity_t050_persist"], lw=0.8,
                 color="tab:green", label="policy_intensity_t050 (persist)")
    axes[2].set_ylabel("policy_intensity")
    axes[2].set_title("Policy-stance intensity (share of sentences classified hawkish or dovish)")
    annotate_events(axes[2])

    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xlabel("date")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOC-LEVEL DISTRIBUTIONS BY DOC_TYPE
# ─────────────────────────────────────────────────────────────────────────────

def plot_doc_distributions(doc, out):
    """Boxplots of P_doc / N_doc / intensity by doc_type."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    doc_types = ["monetary_policy_statement", "monthly_bulletin",
                 "economic_bulletin", "speech"]

    metrics = [
        ("P_doc_dovish_t050",   "P_doc dovish",   axes[0]),
        ("P_doc_hawkish_t050",  "P_doc hawkish",  axes[1]),
        ("policy_intensity_t050", "policy_intensity", axes[2]),
    ]
    for col, title, ax in metrics:
        data = [doc[doc["doc_type"] == t][col].dropna().values for t in doc_types]
        ax.boxplot(data, labels=[t.replace("_", " ") for t in doc_types],
                   showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCATTER P_doc vs N_doc (correlation structure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(doc, out):
    """Scatter of P_doc_dovish vs P_doc_hawkish, by doc_type, with intensity diagonal."""
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = {
        "monetary_policy_statement": "tab:red",
        "monthly_bulletin":          "tab:purple",
        "economic_bulletin":         "tab:orange",
        "speech":                    "tab:blue",
    }
    for t, c in palette.items():
        sub = doc[doc["doc_type"] == t]
        ax.scatter(sub["P_doc_dovish_t050"], sub["P_doc_hawkish_t050"],
                   s=12, alpha=0.4, color=c, label=t.replace("_", " "))

    # Equality line
    lim = max(doc["P_doc_dovish_t050"].max(), doc["P_doc_hawkish_t050"].max())
    ax.plot([0, lim], [0, lim], "k--", lw=0.5, alpha=0.5, label="P=H (no asymmetry)")
    ax.set_xlabel("P_doc dovish")
    ax.set_ylabel("P_doc hawkish")
    corr = doc["P_doc_dovish_t050"].corr(doc["P_doc_hawkish_t050"])
    ax.set_title(f"Doc-level dovish vs hawkish — corr = {corr:+.3f}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANNUAL STRUCTURAL CHECK
# ─────────────────────────────────────────────────────────────────────────────

def plot_annual_means(doc, out):
    """Mean P_doc / N_doc / intensity per year, all sources."""
    doc = doc.copy()
    doc["pub_date"] = pd.to_datetime(doc["pub_date"], errors="coerce")
    doc = doc.dropna(subset=["pub_date"])
    doc["year"] = doc["pub_date"].dt.year
    by_year = doc.groupby("year")[
        ["P_doc_dovish_t050", "P_doc_hawkish_t050", "policy_intensity_t050"]
    ].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(by_year.index, by_year["P_doc_dovish_t050"], "-o", label="dovish (mean)", color="tab:blue")
    ax.plot(by_year.index, by_year["P_doc_hawkish_t050"], "-o", label="hawkish (mean)", color="tab:red")
    ax.plot(by_year.index, by_year["policy_intensity_t050"], "-o", label="intensity (mean)", color="tab:green")
    ax.axvline(2018.5, ls="--", color="gray", alpha=0.5)
    ax.text(2018.6, ax.get_ylim()[1]*0.95, "in-sample / out-of-sample",
            color="gray", fontsize=8)
    ax.set_xlabel("year")
    ax.set_ylabel("mean across documents in year")
    ax.set_title("Annual structural check — mean stance scores by year, all sources")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SENTENCE-LEVEL CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(sent, out):
    """Stacked-bar share of dovish/hawkish/neutral sentences by doc_type."""
    s = sent.copy()
    probs = s[["dovish_score", "hawkish_score", "neutral_score"]].to_numpy()
    classes = np.array(["dovish", "hawkish", "neutral"])
    s["max_class"] = classes[probs.argmax(axis=1)]

    table = (
        s.groupby(["doc_type", "max_class"]).size()
         .unstack(fill_value=0)
    )
    table = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    table[["dovish", "hawkish", "neutral"]].plot(
        kind="bar", stacked=True,
        color=["tab:blue", "tab:red", "lightgrey"],
        ax=ax,
    )
    ax.set_ylabel("share of sentences")
    ax.set_title("FOMC-RoBERTa sentence-level argmax class share by doc_type")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. THRESHOLD SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_sensitivity(daily, out):
    """How does the daily P_t / N_t curve change with threshold choice?"""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    d = daily.set_index("date")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for label, color in [("t000","tab:blue"), ("t050","tab:orange"), ("t080","tab:green")]:
        axes[0].plot(d.index, d[f"P_t_{label}"], lw=0.6, alpha=0.8, label=label, color=color)
        axes[1].plot(d.index, d[f"N_t_{label}"], lw=0.6, alpha=0.8, label=label, color=color)
    axes[0].set_title("Threshold sensitivity — P_t (dovish, rescaled)")
    axes[1].set_title("Threshold sensitivity — N_t (hawkish, rescaled)")
    axes[0].set_ylabel("P_t"); axes[1].set_ylabel("N_t")
    axes[0].legend(); axes[1].legend()
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY STATS TO TEXT FILE
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(sent, doc, daily, out):
    lines = []
    lines.append("DIAGNOSTICS V4 — SUMMARY STATISTICS\n" + "=" * 50)

    lines.append(f"\nSentences: {len(sent):,}")
    lines.append(f"Documents: {sent['doc_id'].nunique():,}")
    by_type = sent["doc_type"].value_counts().to_dict()
    lines.append(f"  by doc_type: {by_type}")

    lines.append("\nDoc-level summary (t050 threshold):")
    desc = doc[[
        "P_doc_dovish_t050", "P_doc_hawkish_t050",
        "policy_intensity_t050", "n_total"
    ]].describe(percentiles=[.5,.75,.95]).round(3)
    lines.append(desc.to_string())

    lines.append("\nDoc-level corr matrix (t050):")
    corr = doc[[
        "P_doc_dovish_t050", "P_doc_hawkish_t050",
        "policy_intensity_t050",
    ]].corr().round(3)
    lines.append(corr.to_string())

    lines.append("\nMean stance by doc_type (t050):")
    mn = doc.groupby("doc_type")[
        ["P_doc_dovish_t050","P_doc_hawkish_t050","policy_intensity_t050","n_total"]
    ].mean().round(3)
    lines.append(mn.to_string())

    lines.append("\nDaily series @t050 (Sadik-rescaled, persist):")
    cols = ["P_t_t050", "N_t_t050", "S_t_t050", "policy_intensity_t050_persist"]
    desc = daily[cols].describe(percentiles=[.05,.5,.95]).round(3)
    lines.append(desc.to_string())

    # Time-series ACF at lag-1 of stance — a quick persistence check
    lines.append("\nDaily series autocorrelation (lag 1, on event-day-only series):")
    eo = daily[daily["is_event_day"] == 1]
    for c in ["P_doc_dovish_t050", "P_doc_hawkish_t050", "policy_intensity_t050"]:
        s = eo[c].dropna()
        ac = s.autocorr(1) if len(s) > 1 else float("nan")
        lines.append(f"  {c:<35} ACF(1) = {ac:+.3f}")

    out.write_text("\n".join(lines))
    log.info(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("DIAGNOSTICS V4 — generating plots and summary")
    log.info("=" * 70)

    log.info("Loading inputs ...")
    sent  = pd.read_csv(SENT_CSV)
    doc   = pd.read_csv(DOC_CSV)
    daily = pd.read_csv(DAILY_ALL)
    log.info(f"  sentences: {len(sent):,}  docs: {len(doc):,}  days: {len(daily):,}")

    plot_daily_series(daily, DIAG_DIR / "01_daily_series.png")
    plot_doc_distributions(doc, DIAG_DIR / "02_doc_distributions_by_type.png")
    plot_scatter(doc, DIAG_DIR / "03_scatter_dovish_vs_hawkish.png")
    plot_annual_means(doc, DIAG_DIR / "04_annual_means.png")
    plot_class_distribution(sent, DIAG_DIR / "05_sentence_class_distribution.png")
    plot_threshold_sensitivity(daily, DIAG_DIR / "06_threshold_sensitivity.png")
    write_summary(sent, doc, daily, DIAG_DIR / "summary.txt")

    log.info("=" * 70)
    log.info(f"DIAGNOSTICS COMPLETE — outputs in {DIAG_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
