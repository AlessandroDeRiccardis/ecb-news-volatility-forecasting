"""
extract_surprises_bernoth.py — Bernoth-style stance-surprise extraction
======================================================================

Adapts Bernoth (2026, DIW Discussion Paper, "Dovish Coos or Hawkish Screech?")
to our daily NA-GARCH framework. Bernoth residualizes a meeting-frequency stance
indicator on macro-financial controls and AR lags, then interprets the residual as
a "communication shock". We adapt this to event-day frequency across all ECB
communication types (MPS, bulletins, speeches), using daily financial-state
controls (true monthly macro variables aren't available at daily frequency).

PROCEDURE
---------
1. Load model_data_master.csv (existing dataset; has P_t, N_t, log_return,
   vstoxx, yield_10y_aaa, period flags).
2. Optionally merge new control series from market_data/ (2y Bund yield, EUR/USD,
   VIX, etc.) — see CONTROL_FILES below.
3. Detect event days as days on which the persisted stance series moved
   (i.e. a new ECB document was published).
4. First-stage OLS on event days within the IN-SAMPLE window (2008–2018):
       P_t = α + Σ_k γ_k · controls_t + φ_1 P_{t-1} + φ_2 P_{t-2} + ε_t
       N_t = α + Σ_k γ_k · controls_t + φ_1 N_{t-1} + φ_2 N_{t-2} + ε_t
   (lags are taken on the persisted series, i.e. the latest available stance
   value as of the previous trading day — valid for OOS use.)
5. Apply estimated coefficients to compute residuals on event days across the
   FULL sample (in-sample + OOS). No re-estimation OOS — coefficients fixed at
   their in-sample values, which is forecast-compatible.
6. Truncate to Sadik supports:
       P_bernoth = max(0, residual_P)        (positive dovish surprises only)
       N_bernoth = min(0, residual_N)        (negative hawkish-side residuals only)
   Rationale: NA-GARCH-asym expects P ≥ 0 and N ≤ 0; signed residuals would
   let the variance scaling drop below the no-news baseline, which diverges
   from Sadik's spec. Truncation preserves the "news weakly raises variance"
   property at the cost of zeroing out wrong-sign deviations.
7. Forward-fill (persist-to-next-event) to produce a daily series.
8. Sadik in-sample-max rescaling so series lie in [0,1] / [-1,0] in-sample.
9. Save as P_t_bernoth, N_t_bernoth in model_data_master_bernoth.csv.
10. Re-fit NAGarchAsym (B2.2 = levels) and NAGarchAsym (B2.4 = bernoth surprises)
    in-sample; print comparison vs. existing B2.2/B2.3 results.

OUTPUT
------
    output/model_data_master_bernoth.csv     — master + bernoth surprise columns
    output/in_sample_summary_bernoth.txt     — comparison vs B2.2/B2.3
    output/extract_surprises_bernoth.log

USAGE
-----
    python extract_surprises_bernoth.py
    python extract_surprises_bernoth.py --controls vstoxx,log_return,yield_10y_aaa,yield_2y_aaa,eur_usd,vix
    python extract_surprises_bernoth.py --no-stance-lags
    python extract_surprises_bernoth.py --oos        # quick OOS h=1 (slow)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models import NAGarchAsym


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./output")
MARKET_DIR = Path("./market_data")
MASTER_CSV = OUTPUT_DIR / "model_data_master.csv"
MASTER_OUT = OUTPUT_DIR / "model_data_master_bernoth.csv"
SUMMARY_OUT = OUTPUT_DIR / "in_sample_summary_bernoth.txt"

# Optional control CSVs to merge in (each: 2 columns, "date" + "<colname>").
# Add new files here as you obtain them. Missing files are silently skipped
# with a warning, so you can run before all files are downloaded.
EXTRA_CONTROL_FILES = {
    "yield_1y_aaa":  MARKET_DIR / "ecb_aaa_1y_yield.csv",   # short end of ECB AAA curve
    "eur_usd":       MARKET_DIR / "eur_usd_daily.csv",
    "vix":           MARKET_DIR / "vix_daily.csv",
    "brent":         MARKET_DIR / "brent_daily.csv",
    "short_rate":    MARKET_DIR / "short_rate_daily.csv",
}

# Default control set used in the first-stage regression.
# Anything not present in the merged dataframe is silently dropped.
DEFAULT_CONTROLS = [
    "log_return",
    "vstoxx",
    "yield_10y_aaa",
    "yield_1y_aaa",
    "eur_usd",
    "vix",
    "brent",
]

# Distribution for NA-GARCH innovations
DIST = "studentst"

# Minimum number of event days required in-sample to estimate first stage
MIN_EVENTS_IS = 50


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "extract_surprises_bernoth.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def load_master():
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"{MASTER_CSV} not found. Run prepare_master_dataset_v4.py first.")
    df = pd.read_csv(MASTER_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    log.info(f"  Master loaded: {len(df):,} rows from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def merge_extra_controls(df, files):
    """Merge any available extra control CSVs into the master dataframe."""
    for col, path in files.items():
        if col in df.columns:
            log.info(f"  '{col}' already in master, skipping merge from {path.name}.")
            continue
        if not path.exists():
            log.warning(f"  Control file '{path.name}' not found — skipping. "
                        f"Add it to market_data/ to include in regression.")
            continue
        ctrl = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        # Auto-detect the value column if user named it something other than `col`
        value_cols = [c for c in ctrl.columns if c != "date"]
        if len(value_cols) == 0:
            log.warning(f"  '{path.name}' has no value column — skipping.")
            continue
        if col not in ctrl.columns:
            ctrl = ctrl.rename(columns={value_cols[0]: col})
        df = df.merge(ctrl[["date", col]], on="date", how="left")
        log.info(f"  Merged '{col}' from {path.name} ({ctrl[col].notna().sum():,} non-NaN obs).")
    return df


def detect_event_days(df):
    """Mark event days = days the persisted stance moved relative to the
    previous trading day. This identifies trading days on which an ECB
    document released news (counting weekend documents allocated to Mon)."""
    p_diff = df["P_t"].fillna(0).diff().abs()
    n_diff = df["N_t"].fillna(0).diff().abs()
    df = df.copy()
    df["is_event_day"] = ((p_diff > 1e-12) | (n_diff > 1e-12)).fillna(False).astype(int)
    log.info(f"  Event days detected: {int(df['is_event_day'].sum()):,} of {len(df):,} trading days")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIRST-STAGE RESIDUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _ols_with_summary(X, y, name):
    """Plain OLS with a few diagnostic stats. Returns coefficients and R²."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    log.info(f"    [{name}] n={len(y)}, k={X.shape[1]}, R²={r2:.4f}")
    return beta, r2


def residualize(df, controls, use_stance_lags=True):
    """
    Run first-stage OLS for P_t and N_t on event days within the in-sample
    window. Apply estimated β across the full sample to compute residuals on
    every event day (in-sample + OOS).

    Returns a copy of df with columns:
        P_resid, N_resid           (raw OLS residuals on event days; NaN otherwise)
    """
    df = df.copy()

    # Lagged stance (persisted) — available on day t-1 for forecast purposes
    df["P_lag1"] = df["P_t"].shift(1)
    df["P_lag2"] = df["P_t"].shift(2)
    df["N_lag1"] = df["N_t"].shift(1)
    df["N_lag2"] = df["N_t"].shift(2)

    available = [c for c in controls if c in df.columns]
    missing = [c for c in controls if c not in df.columns]
    if missing:
        log.warning(f"  Controls not in master, dropping: {missing}")
    log.info(f"  Controls used: {available}")

    feature_cols_P = available + (["P_lag1", "P_lag2"] if use_stance_lags else [])
    feature_cols_N = available + (["N_lag1", "N_lag2"] if use_stance_lags else [])

    # Drop rows with any NaN in features or target
    is_event = df["is_event_day"] == 1
    is_in_sample = df["period"] == "insample"

    # ── P regression ────────────────────────────────────────────────────────
    df["_P_target_present"] = df["P_t"].notna() & df[feature_cols_P].notna().all(axis=1)
    train_mask_P = is_event & is_in_sample & df["_P_target_present"]
    if train_mask_P.sum() < MIN_EVENTS_IS:
        raise RuntimeError(
            f"Only {train_mask_P.sum()} usable in-sample event days for P regression "
            f"(need ≥{MIN_EVENTS_IS}). Check controls / event detection."
        )
    X_P_train = np.column_stack([
        np.ones(train_mask_P.sum()),
        df.loc[train_mask_P, feature_cols_P].to_numpy(dtype=float),
    ])
    y_P_train = df.loc[train_mask_P, "P_t"].to_numpy(dtype=float)
    log.info(f"\n  First-stage regression: P_t (dovish)")
    beta_P, r2_P = _ols_with_summary(X_P_train, y_P_train, "P first-stage")

    # Apply to all event days where features are available
    full_mask_P = is_event & df["_P_target_present"]
    X_P_full = np.column_stack([
        np.ones(full_mask_P.sum()),
        df.loc[full_mask_P, feature_cols_P].to_numpy(dtype=float),
    ])
    yhat_P = X_P_full @ beta_P
    df["P_resid"] = np.nan
    df.loc[full_mask_P, "P_resid"] = (df.loc[full_mask_P, "P_t"].to_numpy(dtype=float) - yhat_P)

    # ── N regression ────────────────────────────────────────────────────────
    df["_N_target_present"] = df["N_t"].notna() & df[feature_cols_N].notna().all(axis=1)
    train_mask_N = is_event & is_in_sample & df["_N_target_present"]
    X_N_train = np.column_stack([
        np.ones(train_mask_N.sum()),
        df.loc[train_mask_N, feature_cols_N].to_numpy(dtype=float),
    ])
    y_N_train = df.loc[train_mask_N, "N_t"].to_numpy(dtype=float)
    log.info(f"\n  First-stage regression: N_t (hawkish)")
    beta_N, r2_N = _ols_with_summary(X_N_train, y_N_train, "N first-stage")

    full_mask_N = is_event & df["_N_target_present"]
    X_N_full = np.column_stack([
        np.ones(full_mask_N.sum()),
        df.loc[full_mask_N, feature_cols_N].to_numpy(dtype=float),
    ])
    yhat_N = X_N_full @ beta_N
    df["N_resid"] = np.nan
    df.loc[full_mask_N, "N_resid"] = (df.loc[full_mask_N, "N_t"].to_numpy(dtype=float) - yhat_N)

    # Persist diagnostics
    df.attrs["bernoth_first_stage"] = {
        "feature_cols_P": feature_cols_P,
        "feature_cols_N": feature_cols_N,
        "beta_P": beta_P.tolist(),
        "beta_N": beta_N.tolist(),
        "r2_P":   r2_P,
        "r2_N":   r2_N,
        "n_train_P": int(train_mask_P.sum()),
        "n_train_N": int(train_mask_N.sum()),
    }

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. POST-PROCESSING (truncate, persist, rescale)
# ─────────────────────────────────────────────────────────────────────────────

def build_bernoth_series(df):
    """
    Convert raw event-day residuals into daily P_t_bernoth, N_t_bernoth series:
        - truncate to Sadik supports
        - forward-fill from event days to all trading days
        - in-sample-max rescale so in-sample range is [0, 1] / [-1, 0]
    """
    df = df.copy()

    # 1. Truncate. Dovish surprise = max(0, P_resid). Hawkish surprise = min(0, N_resid).
    df["P_bernoth_event"] = df["P_resid"].clip(lower=0)
    df["N_bernoth_event"] = df["N_resid"].clip(upper=0)

    # 2. Forward-fill to non-event trading days (persist-to-next-event).
    # Note: the residual is only defined on event days; between events we hold
    # the most-recent-event surprise. Days before the first event remain NaN.
    df["P_bernoth_persist"] = df["P_bernoth_event"].ffill()
    df["N_bernoth_persist"] = df["N_bernoth_event"].ffill()

    # 3. Sadik in-sample-max rescaling.
    is_in_sample = df["period"] == "insample"
    P_is_max = df.loc[is_in_sample, "P_bernoth_persist"].max(skipna=True)
    N_is_min = df.loc[is_in_sample, "N_bernoth_persist"].min(skipna=True)

    P_div = P_is_max if (P_is_max is not None and P_is_max > 0) else 1.0
    N_div = abs(N_is_min) if (N_is_min is not None and N_is_min < 0) else 1.0

    df["P_t_bernoth"] = df["P_bernoth_persist"] / P_div
    df["N_t_bernoth"] = df["N_bernoth_persist"] / N_div  # already ≤ 0

    df.attrs["bernoth_rescaling"] = {"P_is_max": float(P_is_max), "N_is_min": float(N_is_min)}

    # Diagnostics
    n_pos_dovish = int((df["P_bernoth_event"] > 0).sum())
    n_neg_hawkish = int((df["N_bernoth_event"] < 0).sum())
    n_events = int(df["is_event_day"].sum())
    log.info(f"\n  Truncation diagnostics:")
    log.info(f"    Event days with positive dovish surprise:  {n_pos_dovish:,} / {n_events:,} "
             f"({100*n_pos_dovish/max(n_events,1):.1f}%)")
    log.info(f"    Event days with negative hawkish surprise: {n_neg_hawkish:,} / {n_events:,} "
             f"({100*n_neg_hawkish/max(n_events,1):.1f}%)")
    log.info(f"  In-sample max(P_bernoth) = {P_is_max:.6f},  min(N_bernoth) = {N_is_min:.6f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. RE-ESTIMATION (in-sample comparison)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_nagarch(returns, P, N, label):
    log.info(f"  Fitting {label} ...")
    m = NAGarchAsym(returns, P=P, N=N, dist=DIST).fit()
    log.info(f"    {label}: logL = {m.loglik:.4f}, AIC = {m.aic:.4f}, n_params = {len(m.params)}")
    return m


def estimate_in_sample(df):
    """Re-estimate B2.2 (raw) and a Bernoth-surprise variant; print comparison."""
    insample = df[df["period"] == "insample"].dropna(subset=[
        "log_return", "P_t", "N_t", "P_t_bernoth", "N_t_bernoth"
    ]).reset_index(drop=True)
    log.info(f"\n  In-sample n_obs (after dropna of bernoth columns): {len(insample):,}")

    log.info("\n  ── Estimation ──")
    m_lvl = _fit_nagarch(insample["log_return"],
                         insample["P_t"], insample["N_t"],
                         label="B2.2 NA-GARCH-asym (levels) — recomputed for comparable n")
    m_brn = _fit_nagarch(insample["log_return"],
                         insample["P_t_bernoth"], insample["N_t_bernoth"],
                         label="B2.4 NA-GARCH-asym (bernoth surprises)")

    return m_lvl, m_brn, insample


def write_summary(m_lvl, m_brn, df, insample, args):
    diag = df.attrs.get("bernoth_first_stage", {})
    rsc = df.attrs.get("bernoth_rescaling", {})
    feature_cols_P = diag.get("feature_cols_P", [])
    lines = []
    lines.append("BERNOTH-STYLE STANCE-SURPRISE EXTENSION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("First-stage OLS (in-sample event days only)")
    lines.append("-" * 70)
    lines.append(f"  Controls used:       {feature_cols_P}")
    lines.append(f"  Stance lags included:{not args.no_stance_lags}")
    lines.append(f"  n events in-sample:  P={diag.get('n_train_P','-')}, N={diag.get('n_train_N','-')}")
    lines.append(f"  R² (P regression):   {diag.get('r2_P', float('nan')):.4f}")
    lines.append(f"  R² (N regression):   {diag.get('r2_N', float('nan')):.4f}")
    lines.append(f"  In-sample max P_resid_truncated:  {rsc.get('P_is_max', float('nan')):.6f}")
    lines.append(f"  In-sample min N_resid_truncated:  {rsc.get('N_is_min', float('nan')):.6f}")
    lines.append("")
    lines.append("In-sample NA-GARCH-asym estimates (n_obs = {:,})".format(len(insample)))
    lines.append("-" * 70)
    lines.append(f"  B2.2 levels:               logL = {m_lvl.loglik:>10.4f}   AIC = {m_lvl.aic:>10.4f}")
    lines.append(f"  B2.4 bernoth surprises:    logL = {m_brn.loglik:>10.4f}   AIC = {m_brn.aic:>10.4f}")
    lines.append(f"  ΔLL (bernoth − levels):    {m_brn.loglik - m_lvl.loglik:+.4f}")
    lines.append(f"  ΔAIC (bernoth − levels):   {m_brn.aic    - m_lvl.aic:+.4f}")
    lines.append("")
    lines.append("Parameter estimates")
    lines.append("-" * 70)
    lines.append("  B2.2 levels:")
    for k, v in m_lvl.params.items():
        lines.append(f"    {k:>12s} = {v:+.6f}")
    lines.append("  B2.4 bernoth:")
    for k, v in m_brn.params.items():
        lines.append(f"    {k:>12s} = {v:+.6f}")
    lines.append("")
    lines.append("Reference (from in_sample_summary.txt — full-sample fits):")
    lines.append("-" * 70)
    lines.append("  B1.1 GARCH(1,1):                 logL = 8193.027  AIC = -16378.055")
    lines.append("  B1.2 GJR-GARCH(1,1):             logL = 8253.447  AIC = -16496.894")
    lines.append("  B1.3 EGARCH(1,1):                logL = 8270.714  AIC = -16531.429")
    lines.append("  B2.2 NA-GARCH-asym (levels):     logL = 8195.216  AIC = -16374.432")
    lines.append("  B2.3 NA-GARCH-asym (MPS surpr.): logL = 8195.185  AIC = -16374.370")
    lines.append("")
    lines.append("DECISION RULE")
    lines.append("-" * 70)
    lines.append("  If ΔLL (bernoth − levels) is small (e.g. < 1) and ΔAIC > 0,")
    lines.append("  the bernoth surprise does not improve fit and is not worth")
    lines.append("  including in the paper. Compare to ΔLL ≈ +0 of B2.3 over B2.2.")
    lines.append("")

    out = "\n".join(lines) + "\n"
    SUMMARY_OUT.write_text(out)
    log.info("\n" + out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", default=None,
                        help="Comma-separated control list. Default: log_return,vstoxx,"
                             "yield_10y_aaa,yield_2y_aaa,eur_usd,vix (any not present in "
                             "merged dataframe are silently dropped).")
    parser.add_argument("--no-stance-lags", action="store_true",
                        help="Drop AR(2) lags of the stance series from the first stage.")
    parser.add_argument("--oos", action="store_true",
                        help="Also run a quick OOS h=1 evaluation (slow).")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("EXTRACT_SURPRISES_BERNOTH")
    log.info("=" * 70)

    # ── 1. Load + merge ────────────────────────────────────────────────────
    df = load_master()
    df = merge_extra_controls(df, EXTRA_CONTROL_FILES)

    # ── 2. Event day detection ─────────────────────────────────────────────
    df = detect_event_days(df)

    # ── 3. First-stage residualization ─────────────────────────────────────
    if args.controls:
        controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    else:
        controls = list(DEFAULT_CONTROLS)
    df = residualize(df, controls=controls, use_stance_lags=not args.no_stance_lags)

    # ── 4. Build bernoth surprise series ───────────────────────────────────
    df = build_bernoth_series(df)

    # ── 5. Save augmented master ───────────────────────────────────────────
    cols_to_keep = [c for c in df.columns if not c.startswith("_")]
    df[cols_to_keep].to_csv(MASTER_OUT, index=False)
    log.info(f"\n  Wrote augmented master → {MASTER_OUT}")

    # ── 6. Re-estimate in-sample, compare ──────────────────────────────────
    m_lvl, m_brn, insample = estimate_in_sample(df)
    write_summary(m_lvl, m_brn, df, insample, args)

    # ── 7. Optional OOS (gated; this is the slow path) ─────────────────────
    if args.oos:
        log.info("\n  --oos requested. Running quick rolling-window OOS h=1 ...")
        run_quick_oos(df)

    log.info("=" * 70)
    log.info("DONE")
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUICK OOS (gated behind --oos; minimal-cost rolling-window h=1)
# ─────────────────────────────────────────────────────────────────────────────

def run_quick_oos(df):
    """
    Minimal RW h=1 OOS using NAGarchAsym with bernoth inputs.
    Re-fits every WEEK_STEP days to keep runtime tractable. This is NOT a
    replacement for the main forecast_oos.py pipeline — it's a sanity check
    so you can see whether the bernoth signal moves the QLIKE needle before
    committing to the full re-estimation.
    """
    WEEK_STEP = 5
    INSAMPLE_LEN = (df["period"] == "insample").sum()
    df_full = df.dropna(subset=["log_return", "P_t_bernoth", "N_t_bernoth"]).reset_index(drop=True)
    oos_idx_start = df_full.index[df_full["period"] == "outsample"].min()
    if oos_idx_start is np.nan or pd.isna(oos_idx_start):
        log.warning("  No outsample rows available; skipping OOS.")
        return

    rows = []
    n_steps = 0
    for origin in range(int(oos_idx_start), len(df_full) - 1, WEEK_STEP):
        train = df_full.iloc[origin - INSAMPLE_LEN: origin].reset_index(drop=True)
        next_row = df_full.iloc[origin]
        try:
            m_lvl = NAGarchAsym(train["log_return"],
                                P=train["P_t"], N=train["N_t"], dist=DIST).fit()
            m_brn = NAGarchAsym(train["log_return"],
                                P=train["P_t_bernoth"], N=train["N_t_bernoth"], dist=DIST).fit()
            sigma2_lvl = m_lvl.forecast_variance(h=1)
            sigma2_brn = m_brn.forecast_variance(h=1)
            sq_actual  = float(next_row["sq_return"])
            rows.append({
                "date":       next_row["date"],
                "sq_actual":  sq_actual,
                "f_lvl":      sigma2_lvl,
                "f_brn":      sigma2_brn,
            })
            n_steps += 1
        except Exception as e:
            log.warning(f"  OOS step at index {origin} failed: {e}")
            continue
        if n_steps % 10 == 0:
            log.info(f"    {n_steps} OOS steps complete")

    if not rows:
        log.warning("  No usable OOS rows.")
        return

    res = pd.DataFrame(rows)
    qlike_lvl = (res["sq_actual"] / res["f_lvl"] - np.log(res["sq_actual"] / res["f_lvl"]) - 1).mean()
    qlike_brn = (res["sq_actual"] / res["f_brn"] - np.log(res["sq_actual"] / res["f_brn"]) - 1).mean()
    log.info(f"\n  Quick OOS (RW, h=1, every {WEEK_STEP} days, n={len(res)}):")
    log.info(f"    QLIKE B2.2 levels:           {qlike_lvl:.6f}")
    log.info(f"    QLIKE B2.4 bernoth:          {qlike_brn:.6f}")
    log.info(f"    Δ (bernoth − levels):        {qlike_brn - qlike_lvl:+.6f}  "
             f"(negative = bernoth better)")


if __name__ == "__main__":
    main()
