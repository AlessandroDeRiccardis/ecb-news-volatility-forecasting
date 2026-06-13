"""
prepare_master_dataset_v4.py — assemble the modeling dataset
============================================================

Joins:
    - market_data/eurostoxx50_returns.csv   (log_return / sq_return / abs_return)
    - market_data/vstoxx_daily.csv          (vstoxx)
    - market_data/ecb_aaa_10y_yield.csv     (yield_10y_aaa)
    - output/sentiment_daily_all_sources_v4.csv   (P_t / N_t / S_t)
    - output/sentiment_daily_mps_only_v4.csv      (P_t_mps / N_t_mps)
    - output/sentiment_daily_speeches_only_v4.csv (P_t_speech / N_t_speech)

Computes:
    - sq_return_5d (h=5 target = sum of next 5 sq_returns)
    - roll_vol_{10,20,60}d (standard deviation of log_return over rolling window)
    - long_run_mean_{10,20,60}d (IN-SAMPLE MEAN of roll_vol_*d, repeated as constant)

Period flags:
    - presample: 2007-04-02 to 2007-12-31  (burn-in only, no stance values)
    - insample : 2008-01-01 to 2018-12-31  (parameter estimation + AIC)
    - outsample: 2019-01-01 to 2023-12-31  (RW/IW evaluation)

Trading-day calendar comes from the Euro Stoxx 50 returns file (Yahoo gives
weekdays where the index trades). Stance values are forward-filled from the
v4 daily series (which uses persist-to-next-event smoothing on calendar
days). After intersecting with trading days, this means a doc published
Saturday/Sunday is automatically reflected in Monday's stance — the
"weekend allocation" is implicit.

Output: output/model_data_master.csv

USAGE
-----
    python prepare_master_dataset_v4.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("./output")
MARKET_DIR  = Path("./market_data")

RETURNS_CSV = MARKET_DIR / "eurostoxx50_returns.csv"
VSTOXX_CSV    = MARKET_DIR / "vstoxx_daily.csv"
YIELD_10Y_CSV = MARKET_DIR / "ecb_aaa_10y_yield.csv"
YIELD_1Y_CSV  = MARKET_DIR / "ecb_aaa_1y_yield.csv"
EURUSD_CSV    = MARKET_DIR / "eur_usd_daily.csv"
VIX_CSV       = MARKET_DIR / "vix_daily.csv"
BRENT_CSV     = MARKET_DIR / "brent_daily.csv"
DAILY_ALL   = OUTPUT_DIR / "sentiment_daily_all_sources_v4.csv"
DAILY_MPS   = OUTPUT_DIR / "sentiment_daily_mps_only_v4.csv"
DAILY_SPCH  = OUTPUT_DIR / "sentiment_daily_speeches_only_v4.csv"
DAILY_BULL  = OUTPUT_DIR / "sentiment_daily_bulletins_only_v4.csv"

MASTER_CSV  = OUTPUT_DIR / "model_data_master.csv"

# Period boundaries
PRESAMPLE_START = pd.Timestamp("2007-04-01")
INSAMPLE_START  = pd.Timestamp("2008-01-01")
INSAMPLE_END    = pd.Timestamp("2018-12-31")
OUTSAMPLE_END   = pd.Timestamp("2023-12-31")

# Rolling-vol windows (used both as candidates for H_t and as the target
# windows in robustness B5.5).
VOL_WINDOWS = [10, 20, 60]

# Stance "surprise" rolling-mean window (forecast-compatible: deviations
# from the previous W-day mean using only data through t-1). Kept for
# backward compatibility / comparison.
SURPRISE_WINDOW = 20

# Event-driven MPS surprise: at each MPS event, the surprise is the change
# in stance vs the previous MPS event. Between events, the surprise decays
# exponentially with the given half-life (in trading days). 5 days ≈ one
# week, consistent with typical news-impact decay literature; matches
# Sadik et al. (2018) Eq 3.1.
MPS_SURPRISE_HALF_LIFE = 5


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "prepare_master_dataset_v4.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RETURNS + ROLLING VOL
# ─────────────────────────────────────────────────────────────────────────────

def load_returns():
    """Read returns; the trading-day calendar is whatever dates appear here."""
    log.info("\n── Returns + rolling vol ──")
    df = pd.read_csv(RETURNS_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info(f"  {len(df):,} trading days from {df['date'].min().date()} to "
             f"{df['date'].max().date()}")
    return df


def add_rolling_vol_and_long_run_means(df):
    """
    Rolling vol = std(log_return) over the last N trading days.
    long_run_mean = mean of roll_vol_Nd computed ONLY over the in-sample window
    (2008-2018), then propagated as a constant to all rows.
    """
    in_sample_mask = (df["date"] >= INSAMPLE_START) & (df["date"] <= INSAMPLE_END)

    for w in VOL_WINDOWS:
        col_vol = f"roll_vol_{w}d"
        df[col_vol] = df["log_return"].rolling(w, min_periods=w).std()

        in_sample_mean = df.loc[in_sample_mask, col_vol].mean(skipna=True)
        df[f"long_run_mean_{w}d"] = in_sample_mean
        log.info(f"  long_run_mean_{w}d  = {in_sample_mean:.6f}  "
                 f"(in-sample mean of roll_vol_{w}d, used as constant for H_t)")

    return df


def add_h5_target(df):
    """
    h=5 forecast target: sum of next 5 days' sq_return, indexed to today.
    `sq_return_5d` at row t equals sum(sq_return[t+0 ... t+4]).

        sq_return_5d = sq_return.rolling(5).sum().shift(-4)

    The last 4 rows of the series will be NaN (no full 5-day window ahead).
    """
    df["sq_return_5d"] = df["sq_return"].rolling(5).sum().shift(-4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. STANCE / CONTROL JOINS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_join_stance(df):
    log.info("\n── Stance series joins ──")

    daily_all = pd.read_csv(DAILY_ALL, parse_dates=["date"])
    log.info(f"  all-sources daily v4: {len(daily_all):,} calendar days")

    daily_mps = pd.read_csv(DAILY_MPS, parse_dates=["date"])
    daily_spch = pd.read_csv(DAILY_SPCH, parse_dates=["date"])
    daily_bull = pd.read_csv(DAILY_BULL, parse_dates=["date"]) if DAILY_BULL.exists() else None
    if daily_bull is None:
        log.warning(f"  {DAILY_BULL.name} not found — bulletins-only stance will be NaN. "
                    f"Re-run build_daily_series.py to produce it.")

    # Use the t050 / Sadik-rescaled columns produced by build_daily_series.py:
    #   P_t_t050, N_t_t050, S_t_t050
    # These are persist-to-next-event smoothed and rescaled by in-sample max
    # so that in-sample P_t ∈ [0,1], N_t ∈ [-1,0]. (The rescaling step uses
    # in-sample only, so out-of-sample values may exceed those bounds — by
    # design, see Sadik et al. 2018 Eq 3.2.)
    # Also keep τ=0 and τ=0.80 variants from the all-sources file for the
    # threshold-robustness models (B5.6, B5.7).
    all_keep = daily_all[[
        "date",
        "P_t_t050", "N_t_t050", "S_t_t050",
        "P_t_t000", "N_t_t000",
        "P_t_t080", "N_t_t080",
    ]].rename(columns={"P_t_t050": "P_t", "N_t_t050": "N_t", "S_t_t050": "S_t"})
    mps_keep = daily_mps[["date", "P_t_t050", "N_t_t050"]].rename(
        columns={"P_t_t050": "P_t_mps", "N_t_t050": "N_t_mps"}
    )
    spch_keep = daily_spch[["date", "P_t_t050", "N_t_t050"]].rename(
        columns={"P_t_t050": "P_t_speech", "N_t_t050": "N_t_speech"}
    )
    if daily_bull is not None:
        bull_keep = daily_bull[["date", "P_t_t050", "N_t_t050"]].rename(
            columns={"P_t_t050": "P_t_bulletins", "N_t_t050": "N_t_bulletins"}
        )
    else:
        bull_keep = None

    df = df.merge(all_keep,  on="date", how="left")
    df = df.merge(mps_keep,  on="date", how="left")
    df = df.merge(spch_keep, on="date", how="left")
    if bull_keep is not None:
        df = df.merge(bull_keep, on="date", how="left")

    # On trading days within the v4 series window (2008+), there should be a
    # value. Days before 2008 are intentionally NaN (pre-sample, no stance).
    n_in_sample = ((df["date"] >= INSAMPLE_START) & (df["date"] <= INSAMPLE_END)).sum()
    n_in_sample_with_stance = (
        (df["date"] >= INSAMPLE_START) & (df["date"] <= INSAMPLE_END)
        & df["P_t"].notna()
    ).sum()
    log.info(f"  In-sample trading days: {n_in_sample:,}, of which with stance: "
             f"{n_in_sample_with_stance:,} "
             f"({n_in_sample_with_stance/n_in_sample:.1%})")

    return df


def add_stance_surprises(df):
    """
    Forecast-compatible rolling-mean stance surprises:
        X_surprise_t = X_t  −  rolling_mean(X, W=20).shift(1)

    rolling_mean(W).shift(1) uses only data from t-W through t-1
    (strictly past), so X_surprise_t is observable at time t.

    Added for: P_t, N_t, S_t (all-sources). Kept primarily for backward
    compatibility — the methodologically cleaner surprise measure is
    add_mps_event_surprises (event-driven, exponentially-decayed MPS
    surprise; see below).
    """
    log.info("\n── Stance surprises (rolling-mean, all-sources) ──")
    for col in ("P_t", "N_t", "S_t"):
        roll = df[col].rolling(SURPRISE_WINDOW, min_periods=SURPRISE_WINDOW).mean().shift(1)
        df[f"{col[:-2]}_surprise"] = df[col] - roll
    n_in_sample_with = (
        (df["period"] == "insample") & df["P_surprise"].notna()
    ).sum()
    n_in_sample_total = (df["period"] == "insample").sum()
    log.info(f"  In-sample rows with valid surprise: "
             f"{n_in_sample_with:,}/{n_in_sample_total:,} "
             f"(first ~{SURPRISE_WINDOW} dropped due to rolling-mean warm-up)")
    return df


def add_mps_event_surprises(df, half_life_days=MPS_SURPRISE_HALF_LIFE):
    """
    INTRA-PERIOD-BASELINE MPS surprises with exponential decay.

    For each MPS event at date t_mps, the surprise is the deviation of the
    actual MPS stance from the *expected* stance heading into that MPS,
    where the expectation is built from the ECB's intra-period non-MPS
    communications (speeches and bulletins) since the previous MPS.

    More formally, for an MPS event at t_mps with previous MPS at t_prev:

        Let I = {non-MPS docs at dates in (t_prev, t_mps)}
        For each i ∈ I, weight w_i = exp(−λ · (t_mps − t_i))   [days, half-life λ]
        P_expected = Σ_i w_i · P_doc_dovish_i  /  Σ_i w_i
        N_expected = Σ_i w_i · (−P_doc_hawkish_i)  /  Σ_i w_i

        ΔP_mps_event = P_doc_dovish_mps   − P_expected
        ΔN_mps_event = (−P_doc_hawkish_mps) − N_expected

    More-recent intra-period communications get higher weight (same λ as the
    post-event decay below; ~5-day half-life means a speech 2 days before the
    MPS gets ~75% weight, a speech 30 days before gets ~1.5% weight). Fallback
    when no intra-period non-MPS docs exist: ΔX = X_mps − X_prev_mps (i.e., the
    older MPS-to-MPS difference behavior — degenerate but rare).

    ECONOMIC JUSTIFICATION
    ----------------------
    A raw MPS-to-MPS difference treats *anticipated* changes (where economic
    conditions evolved between events and the ECB responded as expected) as
    surprises. The intra-period-baseline approach instead measures the deviation
    of the actual MPS from the picture the ECB had been painting in its other
    communications — analogous to the high-frequency-identification literature
    (Kuttner 2001; Gürkaynak-Sack-Swanson 2005) that uses market prices in a
    tight window around the announcement to extract the surprise component,
    but adapted to a text-based context where market-priced expectations
    aren't available.

    Between MPS events, the surprise decays exponentially (markets digest
    news over ~one trading week), and we pass absolute decayed surprises
    to the NA-GARCH model (intensity reading; any deviation amplifies vol).

    Adds columns:
        dP_mps_event           — signed surprise on MPS event days (else NaN)
        dN_mps_event
        dP_mps_decayed         — signed decayed surprise, daily
        dN_mps_decayed
        P_mps_surprise         — |dP_mps_decayed|   (≥0; model P input)
        N_mps_surprise         — −|dN_mps_decayed|  (≤0; model N input)
        days_since_mps_event   — trading days since most recent MPS event
        n_intra_docs           — count of intra-period non-MPS docs used per event
    """
    log.info("\n── Intra-period-baseline MPS surprises with exponential decay ──")
    log.info(f"  half-life: {half_life_days} trading days "
             f"(λ = {np.log(2)/half_life_days:.4f})")
    log.info(f"  expected stance = decay-weighted mean of non-MPS docs "
             f"between MPS events")

    # Per-doc stance values — load from sentiment_document_level_v4.csv,
    # which has every document (MPS, speech, monthly bulletin, economic
    # bulletin) with its raw P_doc_dovish_t050 and P_doc_hawkish_t050.
    doc_csv = OUTPUT_DIR / "sentiment_document_level_v4.csv"
    if not doc_csv.exists():
        raise FileNotFoundError(
            f"{doc_csv} not found — run build_daily_series.py first.")
    docs = pd.read_csv(doc_csv, parse_dates=["pub_date"])
    docs = docs.dropna(subset=["pub_date", "P_doc_dovish_t050", "P_doc_hawkish_t050"])
    docs = docs.sort_values("pub_date").reset_index(drop=True)
    docs["P_raw"] =  docs["P_doc_dovish_t050"]      # P convention: ≥ 0 (dovish share)
    docs["N_raw"] = -docs["P_doc_hawkish_t050"]     # N convention: ≤ 0 (negated hawkish)

    MPS_TYPES = ["monetary_policy_statement", "combined_monetary_policy_statement"]
    mps_docs   = docs[docs["doc_type"].isin(MPS_TYPES)].copy().reset_index(drop=True)
    intra_docs = docs[~docs["doc_type"].isin(MPS_TYPES)].copy()
    log.info(f"  MPS events: {len(mps_docs):,}, "
             f"non-MPS docs (intra-period candidates): {len(intra_docs):,}")

    lam = float(np.log(2.0) / half_life_days)

    # For each MPS event, compute expected stance and surprise
    surp_records = []
    n_with_intra = n_fallback = 0
    for i, mps in mps_docs.iterrows():
        t_mps = mps["pub_date"]
        if i == 0:
            # First MPS in the corpus: no prior, set surprise to 0
            surp_records.append({
                "date": t_mps, "dP_mps_event": 0.0, "dN_mps_event": 0.0,
                "n_intra_docs": 0,
            })
            continue

        t_prev = mps_docs.iloc[i - 1]["pub_date"]

        # Intra-period non-MPS docs: dates strictly between t_prev and t_mps
        mask = (intra_docs["pub_date"] > t_prev) & (intra_docs["pub_date"] < t_mps)
        intra = intra_docs[mask]

        if len(intra) > 0:
            # Decay-weighted mean expectation. Using calendar days here as a
            # close approximation to trading days; off by at most one weekend
            # which is negligible at half-life 5.
            days_before = (t_mps - intra["pub_date"]).dt.days.values.astype(float)
            w = np.exp(-lam * days_before)
            w = w / w.sum()
            P_expected = float(np.sum(w * intra["P_raw"].values))
            N_expected = float(np.sum(w * intra["N_raw"].values))
            n_with_intra += 1
        else:
            # Fallback: use previous MPS as the expectation (degenerates to
            # MPS-to-MPS difference for this event only). Rare in practice.
            P_expected = float(mps_docs.iloc[i - 1]["P_raw"])
            N_expected = float(mps_docs.iloc[i - 1]["N_raw"])
            n_fallback += 1

        dP = float(mps["P_raw"]) - P_expected
        dN = float(mps["N_raw"]) - N_expected
        surp_records.append({
            "date": t_mps, "dP_mps_event": dP, "dN_mps_event": dN,
            "n_intra_docs": len(intra),
        })

    surp_df = pd.DataFrame(surp_records)
    log.info(f"  MPS events with intra-period docs: {n_with_intra}/{len(mps_docs)} "
             f"({n_with_intra/max(len(mps_docs),1):.0%})")
    if n_fallback > 0:
        log.info(f"  Fallback to MPS-to-MPS (no intra docs): {n_fallback} events")
    log.info(f"  Mean |ΔP_mps_event|: {surp_df['dP_mps_event'].abs().mean():.4f}, "
             f"max: {surp_df['dP_mps_event'].abs().max():.4f}")
    log.info(f"  Mean |ΔN_mps_event|: {surp_df['dN_mps_event'].abs().mean():.4f}, "
             f"max: {surp_df['dN_mps_event'].abs().max():.4f}")

    # Drop any older MPS-surprise columns left over from an earlier prep run
    OLD_COLS = ["dP_mps_event", "dN_mps_event", "dP_mps_decayed", "dN_mps_decayed",
                "P_mps_surprise", "N_mps_surprise", "days_since_mps_event",
                "n_intra_docs"]
    df = df.drop(columns=[c for c in OLD_COLS if c in df.columns])

    # Merge per-event surprise values into the trading-day df by date
    df = df.merge(surp_df, on="date", how="left")

    # Forward-fill across non-event days
    df["dP_persist"] = df["dP_mps_event"].ffill()
    df["dN_persist"] = df["dN_mps_event"].ffill()

    # Trading-day count since most-recent MPS event (block + cumcount)
    df["_block"] = df["dP_mps_event"].notna().cumsum()
    df["days_since_mps_event"] = df.groupby("_block").cumcount()

    # Exponential decay between events
    decay = np.exp(-lam * df["days_since_mps_event"])
    df["dP_mps_decayed"] = (df["dP_persist"] * decay).fillna(0.0)
    df["dN_mps_decayed"] = (df["dN_persist"] * decay).fillna(0.0)

    # Absolute-value inputs for NA-GARCH compatibility (intensity reading)
    df["P_mps_surprise"] =  df["dP_mps_decayed"].abs()
    df["N_mps_surprise"] = -df["dN_mps_decayed"].abs()

    # Clean up intermediates
    df = df.drop(columns=["dP_persist", "dN_persist", "_block"])

    # Diagnostic: in-sample summary
    n_in = (df["period"] == "insample").sum()
    n_with = ((df["period"] == "insample") & (df["P_mps_surprise"] > 0)).sum()
    log.info(f"  In-sample rows with non-zero surprise input: {n_with:,}/{n_in:,} "
             f"({n_with/max(n_in,1):.1%})")
    log.info(f"  Max in-sample |ΔP_mps|: "
             f"{df.loc[df['period']=='insample', 'dP_mps_decayed'].abs().max():.4f}")
    log.info(f"  Max in-sample |ΔN_mps|: "
             f"{df.loc[df['period']=='insample', 'dN_mps_decayed'].abs().max():.4f}")
    return df


def load_and_join_controls(df):
    """
    Merge daily financial-state controls into the master:
      - vstoxx, yield_10y_aaa            (always required)
      - yield_1y_aaa, eur_usd, vix, brent (used for Bernoth-style residualization)

    Missing files are skipped with a warning rather than failing — the master
    is still usable for the main suite without the Bernoth controls.
    """
    log.info("\n── Financial-state controls ──")
    sources = [
        (VSTOXX_CSV,    "vstoxx",        True),
        (YIELD_10Y_CSV, "yield_10y_aaa", True),
        (YIELD_1Y_CSV,  "yield_1y_aaa",  False),
        (EURUSD_CSV,    "eur_usd",       False),
        (VIX_CSV,       "vix",           False),
        (BRENT_CSV,     "brent",         False),
    ]
    for path, col, required in sources:
        if not path.exists():
            msg = f"  Missing {path.name}"
            if required:
                raise FileNotFoundError(f"{msg} (required).")
            log.warning(f"{msg} — skipping; Bernoth surprise extraction will need it.")
            continue
        ctrl = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        # Auto-rename the value column if needed
        value_cols = [c for c in ctrl.columns if c != "date"]
        if not value_cols:
            log.warning(f"  {path.name} has no value column — skipping.")
            continue
        if col not in ctrl.columns:
            ctrl = ctrl.rename(columns={value_cols[0]: col})
        log.info(f"  {col:<14} {len(ctrl):>5,} rows  "
                 f"({ctrl['date'].min().date()} → {ctrl['date'].max().date()})")
        df = df.merge(ctrl[["date", col]], on="date", how="left")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. PERIOD FLAGS
# ─────────────────────────────────────────────────────────────────────────────

def assign_period(df):
    df["period"] = pd.Series(dtype="object")
    df.loc[
        (df["date"] >= PRESAMPLE_START) & (df["date"] < INSAMPLE_START),
        "period"] = "presample"
    df.loc[
        (df["date"] >= INSAMPLE_START) & (df["date"] <= INSAMPLE_END),
        "period"] = "insample"
    df.loc[
        (df["date"] > INSAMPLE_END) & (df["date"] <= OUTSAMPLE_END),
        "period"] = "outsample"
    df = df[df["period"].notna()].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("PREPARE MASTER DATASET v4")
    log.info("=" * 70)

    df = load_returns()
    df = add_h5_target(df)
    df = load_and_join_stance(df)
    df = load_and_join_controls(df)
    df = assign_period(df)
    df = add_mps_event_surprises(df)      # event-driven decayed MPS surprises (B2.3 appendix)

    # Final column order — only what's actually consumed by the modeling pipeline.
    # Dropped:
    #   - roll_vol_*d, long_run_mean_*d  (H_t indicator artifacts; H_t was tested
    #                                     and rejected, see methodology note)
    #   - P_surprise, N_surprise, S_surprise  (legacy rolling-mean surprises;
    #                                          superseded by B2.3 / Bernoth)
    #   - dP_mps_*, dN_mps_*, days_since_mps_event, n_intra_docs
    #     (intermediates for B2.3 build; only the final P_mps_surprise /
    #      N_mps_surprise are consumed by the model)
    cols = [
        "date", "period",
        # Return / target series
        "log_return", "sq_return", "abs_return", "sq_return_5d",
        # All-sources stance (main spec, B2.1 / B2.2)
        "P_t", "N_t", "S_t",
        # Source decomposition (B5.3 / B5.4 / B5.5)
        "P_t_mps",       "N_t_mps",
        "P_t_speech",    "N_t_speech",
        "P_t_bulletins", "N_t_bulletins",
        # Threshold robustness (B5.6 at τ=0; B5.7 at τ=0.80)
        "P_t_t000", "N_t_t000",
        "P_t_t080", "N_t_t080",
        # MPS-vs-intra-period-baseline surprise (B2.3 appendix)
        "P_mps_surprise", "N_mps_surprise",
        # Financial-state controls (Bernoth residualization + general use)
        "vstoxx", "yield_10y_aaa", "yield_1y_aaa",
        "eur_usd", "vix", "brent",
    ]
    # Tolerate missing optional columns (e.g. Bernoth controls not yet downloaded)
    available = [c for c in cols if c in df.columns]
    missing   = [c for c in cols if c not in df.columns]
    if missing:
        log.warning(f"  Master will exclude (not in input data): {missing}")
    df = df[available]

    df.to_csv(MASTER_CSV, index=False)
    log.info(f"\n  Saved → {MASTER_CSV}")
    log.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    log.info("\n  Rows per period:")
    log.info(df["period"].value_counts().to_string())

    # Sanity coverage
    log.info("\n  Coverage of key columns by period:")
    for p in ["presample", "insample", "outsample"]:
        sub = df[df["period"] == p]
        if len(sub) == 0:
            continue
        log.info(f"    {p:<10} ({len(sub):,} rows)")
        for c in ["log_return", "P_t", "vstoxx", "yield_10y_aaa", "yield_1y_aaa", "eur_usd", "vix", "brent"]:
            if c not in sub.columns:
                continue
            n_ok = sub[c].notna().sum()
            log.info(f"      {c:<18}  {n_ok:,}/{len(sub):,}  ({n_ok/len(sub):.1%})")

    log.info("\n  Quick stance sanity (in-sample only):")
    is_df = df[df["period"] == "insample"].copy()
    for c in ["P_t", "N_t", "S_t", "P_t_mps", "N_t_mps", "P_t_speech", "N_t_speech"]:
        s = is_df[c].dropna()
        log.info(f"    {c:<14}  mean={s.mean():+.3f}  std={s.std():.3f}  "
                 f"min={s.min():+.3f}  max={s.max():+.3f}")

    log.info("\n" + "=" * 70)
    log.info("MASTER DATASET COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
