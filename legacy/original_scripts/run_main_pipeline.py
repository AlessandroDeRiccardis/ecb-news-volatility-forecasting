"""
run_main_pipeline.py — final modeling orchestrator
===================================================

Runs the complete in-sample + OOS suite, residual diagnostics, sub-sample OOS
analysis, forecast combination, and economic-significance computation.
Writes one consolidated report (`output/final_results_summary.txt`) plus the
underlying CSVs that back each table.

This is the script teammates run to verify the modeling end-to-end.

PIPELINE
--------
1. Validate inputs (master CSV present, expected columns).
2. Augment master in-memory with bulletins / threshold variants if missing.
3. In-sample MLE for the main suite (B1.1, B1.2, B1.3, B2.1, B2.2).
4. In-sample MLE for the robustness suite (B5.1*, B5.3, B5.4, B5.5, B5.6, B5.7, B5.8).
   * B5.1 is a re-evaluation, not a re-estimation — see step 7.
5. Ljung-Box residual diagnostics on main-suite fits.
6. OOS forecasting (main: RW×IW×h1×h5; source robustness: RW×h1×h5).
7. OOS evaluation: QLIKE/RMSE, DM tests (vs B1.1 and vs B1.2), MCS at 90%,
   sub-sample analysis (2019, 2020–21, 2022–23), abs_return-target re-eval.
8. Forecast combination: optimal w in (1−w) GJR + w NA-GARCH.
9. Economic significance: max/min implied variance scaling under B2.2.
10. Write `output/final_results_summary.txt` consolidating everything.

OPTIONAL APPENDIX
-----------------
B2.3 (MPS-decayed surprises) and the two Bernoth-residualized surprise specs
are run via `--include-appendix`. They are reported in a separate appendix
table, not the main results, because all three underperform raw levels.

USAGE
-----
    python run_main_pipeline.py                   # full run (~2 hours w/ OOS)
    python run_main_pipeline.py --in-sample-only  # ~30 sec sanity check
    python run_main_pipeline.py --skip-oos        # in-sample + diagnostics only
    python run_main_pipeline.py --main-only       # skip robustness rows
    python run_main_pipeline.py --include-appendix
    python run_main_pipeline.py --quick-oos       # 20 OOS origins (smoke test)
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

# Local imports
from models import GARCH11, GJRGARCH, EGARCH, NAGarchAsym, NAGarchNet
from forecast_oos import make_model as make_oos_model
from forecast_oos import run_oos, is_nagarch

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("./output")
MASTER_CSV = OUTPUT_DIR / "model_data_master.csv"
DAILY_ALL  = OUTPUT_DIR / "sentiment_daily_all_sources_v4.csv"
DAILY_BULL = OUTPUT_DIR / "sentiment_daily_bulletins_only_v4.csv"
DOC_CSV    = OUTPUT_DIR / "sentiment_document_level_v4.csv"
OOS_DIR    = OUTPUT_DIR / "oos_forecasts"

DIST = "studentst"

# Model lists
MAIN_MODELS = ["B1.1", "B1.2", "B1.3", "B2.1", "B2.2"]
ROBUSTNESS_MODELS_OOS = ["B5.3", "B5.4", "B5.5"]   # source-decomposition; OOS evaluated
ROBUSTNESS_MODELS_IS_ONLY = ["B5.6", "B5.7"]        # threshold variants; in-sample only
APPENDIX_MODELS = ["B2.3"]                           # surprise variants; in-sample only

OOS_SCHEMES = ["RW", "IW"]
OOS_HORIZONS = [1, 5]
WEEK_STEP = 5

# Sub-sample windows for OOS analysis
SUBSAMPLES = {
    "2019_normal":     ("2019-01-01", "2019-12-31"),
    "2020_2021_covid": ("2020-01-01", "2021-12-31"),
    "2022_2023_infl":  ("2022-01-01", "2023-12-31"),
}

# Ljung-Box lags
LJUNG_BOX_LAGS = [5, 10, 20]


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
OOS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "run_main_pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING + IN-MEMORY AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def load_master():
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"{MASTER_CSV} not found. Run prepare_master_dataset_v4.py first.")
    df = pd.read_csv(MASTER_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    log.info(f"  Loaded master: {len(df):,} rows  "
             f"({df['date'].min().date()} → {df['date'].max().date()})")

    # Required columns for at least the main suite
    required = ["date", "period", "log_return", "sq_return", "sq_return_5d",
                "P_t", "N_t", "S_t"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Master is missing required columns: {missing}")

    return df


def augment_thresholds(df):
    """
    Add P_t_t000, N_t_t000, P_t_t080, N_t_t080 from the all-sources daily
    sentiment file if not already present in master. Used for τ-robustness
    models (B5.6, B5.7).
    """
    needed = ["P_t_t000", "N_t_t000", "P_t_t080", "N_t_t080"]
    present = [c for c in needed if c in df.columns]
    if len(present) == len(needed):
        return df
    if not DAILY_ALL.exists():
        log.warning(f"  {DAILY_ALL} not found — threshold-robustness models will be skipped.")
        return df
    log.info(f"  Augmenting in-memory: threshold variants ({needed})")
    daily = pd.read_csv(DAILY_ALL, parse_dates=["date"])
    keep = [c for c in needed if c in daily.columns]
    df = df.merge(daily[["date"] + keep], on="date", how="left")
    return df


def augment_bulletins(df):
    """
    Add P_t_bulletins, N_t_bulletins from sentiment_daily_bulletins_only_v4.csv
    if not present. Used for B5.5.
    """
    if "P_t_bulletins" in df.columns and "N_t_bulletins" in df.columns:
        return df
    if not DAILY_BULL.exists():
        log.warning(f"  {DAILY_BULL} not found — B5.5 (bulletins-only) will be skipped. "
                    f"Re-run build_daily_series.py to produce it.")
        return df
    log.info(f"  Augmenting in-memory: P_t_bulletins / N_t_bulletins")
    daily = pd.read_csv(DAILY_BULL, parse_dates=["date"])
    keep = daily[["date", "P_t_t050", "N_t_t050"]].rename(
        columns={"P_t_t050": "P_t_bulletins", "N_t_t050": "N_t_bulletins"}
    )
    df = df.merge(keep, on="date", how="left")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. IN-SAMPLE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def fit_in_sample(df, model_name, dist=DIST):
    """Fit one model on the in-sample subset of df; return fitted model."""
    cols_needed = {
        "B1.1": ["log_return"],
        "B1.2": ["log_return"],
        "B1.3": ["log_return"],
        "B2.1": ["log_return", "S_t"],
        "B2.2": ["log_return", "P_t", "N_t"],
        "B2.3": ["log_return", "P_mps_surprise", "N_mps_surprise"],
        "B5.3": ["log_return", "P_t_mps", "N_t_mps"],
        "B5.4": ["log_return", "P_t_speech", "N_t_speech"],
        "B5.5": ["log_return", "P_t_bulletins", "N_t_bulletins"],
        "B5.6": ["log_return", "P_t_t000", "N_t_t000"],
        "B5.7": ["log_return", "P_t_t080", "N_t_t080"],
    }[model_name]

    ins = df[df["period"] == "insample"].dropna(subset=cols_needed).reset_index(drop=True)
    if len(ins) < 100:
        raise RuntimeError(f"{model_name}: only {len(ins)} usable in-sample rows.")

    m = make_oos_model(model_name, ins, dist=dist)
    if is_nagarch(model_name):
        m.fit(n_restarts=2)
    else:
        m.fit()
    return m, ins


def fit_in_sample_suite(df, model_names, dist=DIST):
    """Fit each model in `model_names`. Returns list of dicts and a fits dict."""
    rows = []
    fits = {}
    for name in model_names:
        try:
            t0 = time.time()
            m, ins = fit_in_sample(df, name, dist=dist)
            t = time.time() - t0
            n_params = len(getattr(m, "_theta_opt", m.params)) if is_nagarch(name) else len(m.params)
            rows.append({
                "model":    name,
                "dist":     dist,
                "n_obs":    len(ins),
                "loglik":   m.loglik,
                "aic":      m.aic,
                "n_params": n_params,
                "fit_time": t,
            })
            fits[name] = (m, ins)
            log.info(f"    {name:<6}  n={len(ins):,}  ll={m.loglik:>9.2f}  "
                     f"AIC={m.aic:>10.2f}  ({t:.1f}s)")
        except Exception as e:
            log.error(f"    {name}: FAILED ({e})")
            rows.append({"model": name, "dist": dist, "loglik": np.nan,
                         "aic": np.nan, "n_obs": np.nan, "n_params": np.nan,
                         "fit_time": np.nan})
    return pd.DataFrame(rows), fits


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESIDUAL DIAGNOSTICS (Ljung-Box on standardized residuals)
# ─────────────────────────────────────────────────────────────────────────────

def ljung_box(residuals, lags):
    """
    Ljung-Box Q-statistic with chi-squared p-values.
    Q(L) = T(T+2) Σ_{k=1}^L ρ²_k / (T-k)
    Returns dict {lag: (Q, p, df)}.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < max(lags) + 5:
        return {L: (np.nan, np.nan, L) for L in lags}

    r_centered = r - r.mean()
    denom = float(np.sum(r_centered ** 2))
    if denom <= 0:
        return {L: (np.nan, np.nan, L) for L in lags}

    out = {}
    for L in lags:
        s = 0.0
        for k in range(1, L + 1):
            num = float(np.sum(r_centered[k:] * r_centered[:-k]))
            rho_k = num / denom
            s += (rho_k ** 2) / (T - k)
        Q = T * (T + 2) * s
        p = float(1.0 - chi2.cdf(Q, df=L))
        out[L] = (float(Q), p, L)
    return out


def residual_diagnostics(fits, lags=LJUNG_BOX_LAGS):
    rows = []
    for name, (m, ins) in fits.items():
        try:
            std_resid = m.residuals_standardized()
        except Exception as e:
            log.warning(f"    {name}: cannot compute standardized residuals ({e})")
            continue
        # Q on standardized residuals (autocorrelation in level)
        lb_lvl = ljung_box(std_resid, lags)
        # Q on squared standardized residuals (remaining ARCH effects)
        lb_sq = ljung_box(std_resid ** 2, lags)
        for L in lags:
            Q_lvl, p_lvl, _ = lb_lvl[L]
            Q_sq, p_sq, _ = lb_sq[L]
            rows.append({
                "model": name, "lag": L,
                "Q_resid":    Q_lvl, "p_resid":    p_lvl,
                "Q_resid_sq": Q_sq,  "p_resid_sq": p_sq,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. OOS FORECASTING (delegates to forecast_oos.run_oos)
# ─────────────────────────────────────────────────────────────────────────────

def run_oos_suite(df, model_specs, dist=DIST, week_step=WEEK_STEP, max_origins=None):
    """
    `model_specs` is a list of (model_name, scheme, horizon) tuples.
    Saves each (model, scheme, horizon) result CSV to oos_forecasts/.
    """
    OOS_DIR.mkdir(exist_ok=True)
    written = []
    for model_name, scheme, horizon in model_specs:
        out_path = OOS_DIR / f"{model_name}_{scheme}_h{horizon}.csv"
        log.info(f"  → OOS: {model_name} | {scheme} | h={horizon}  →  {out_path.name}")
        try:
            res = run_oos(df, model_name, scheme, horizon, dist=dist,
                          week_step=week_step, max_origins=max_origins)
            res.to_csv(out_path, index=False)
            written.append(out_path)
        except Exception as e:
            log.error(f"    {model_name} {scheme} h={horizon}: FAILED ({e})")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# 5. OOS EVALUATION: QLIKE, RMSE, DM, MCS, sub-samples
# ─────────────────────────────────────────────────────────────────────────────

def qlike(realized, forecast):
    realized  = np.asarray(realized,  dtype=float)
    forecast  = np.asarray(forecast,  dtype=float)
    eps = 1e-12
    return np.mean(np.log(forecast + eps) + realized / (forecast + eps))


def rmse(realized, forecast):
    realized = np.asarray(realized, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return np.sqrt(np.mean((realized - forecast) ** 2))


def dm_test(loss_a, loss_b, horizon=1):
    """
    Diebold-Mariano test, two-sided, with Newey-West HAC for h>1.
    H0: equal predictive accuracy.
    Returns (mean_diff, DM_stat, p_value). Sign convention: mean_diff = loss_a − loss_b;
    negative → A is better.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 5:
        return np.nan, np.nan, np.nan
    d_bar = d.mean()
    # HAC long-run variance with bandwidth h-1
    bw = max(0, horizon - 1)
    gamma0 = float(np.var(d, ddof=1))
    s = gamma0
    for k in range(1, bw + 1):
        cov_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        s += 2.0 * (1.0 - k / (bw + 1)) * cov_k
    s = max(s, 1e-12)
    DM = d_bar / np.sqrt(s / T)
    p = 2.0 * (1.0 - norm.cdf(abs(DM)))
    return float(d_bar), float(DM), float(p)


def model_confidence_set(losses_dict, alpha=0.10, reps=5000, seed=42):
    """
    Hansen-Lunde-Nason (2011) Model Confidence Set.

    Uses `arch.bootstrap.MCS` (T_max statistic, stationary bootstrap) which is
    the canonical implementation in the volatility-forecasting literature.
    Falls back to a simpler hand-rolled bootstrap only if arch is unavailable
    (which is unlikely since the GARCH benchmarks already require it).

    `losses_dict` = {model_name: array of per-period losses}.
    Returns list of model names retained at confidence (1 − alpha).
    """
    names = list(losses_dict.keys())
    if len(names) < 2:
        return names
    losses_df = pd.DataFrame(losses_dict).dropna(how="any")
    if losses_df.shape[0] < 30:
        return names

    try:
        from arch.bootstrap import MCS
        mcs = MCS(losses_df.values, size=alpha, reps=reps, seed=seed,
                  method="max")  # T_max statistic; standard for vol forecasting
        mcs.compute()
        included_idx = list(mcs.included)
        return [losses_df.columns[i] for i in included_idx]
    except ImportError:
        log.warning("    arch.bootstrap.MCS not available — falling back to "
                    "simpler hand-rolled MCS (less powerful).")
    except Exception as e:
        log.warning(f"    MCS via arch failed ({e}); falling back to hand-rolled.")

    # Fallback: simple stationary-bootstrap MCS
    rng = np.random.default_rng(seed)
    L = losses_df.values.T  # shape (n_models, T)
    T = L.shape[1]
    p_block = 1.0 / max(1.0, T ** (1.0 / 3.0))
    in_set = list(range(len(names)))
    while len(in_set) > 1:
        L_sub = L[in_set]
        d_bar = L_sub - L_sub.mean(axis=0, keepdims=True)
        d_dot = L_sub.mean(axis=1) - L_sub.mean()
        var_d = np.var(d_bar, axis=1, ddof=1) + 1e-12
        t_stat = d_dot / np.sqrt(var_d / T)
        Tmax = float(np.max(t_stat))
        Tmax_b = np.empty(min(reps, 2000))
        for b in range(len(Tmax_b)):
            idx = []
            i = int(rng.integers(0, T))
            while len(idx) < T:
                idx.append(i % T)
                if rng.random() < p_block:
                    i = int(rng.integers(0, T))
                else:
                    i += 1
            Lb = L_sub[:, idx]
            d_b = Lb - Lb.mean(axis=0, keepdims=True)
            d_dot_b = Lb.mean(axis=1) - Lb.mean()
            var_b = np.var(d_b, axis=1, ddof=1) + 1e-12
            Tmax_b[b] = float(np.max(d_dot_b / np.sqrt(var_b / T)))
        p = float(np.mean(Tmax_b >= Tmax))
        if p >= alpha:
            return [names[i] for i in in_set]
        worst_local = int(np.argmax(t_stat))
        in_set.pop(worst_local)
    return [names[in_set[0]]]


def evaluate_oos(model_names, schemes=None, horizons=None,
                 baselines=("B1.1", "B1.2"), subsamples=SUBSAMPLES):
    """
    Build the OOS QLIKE / RMSE / DM / MCS tables across (model, scheme, horizon).
    Returns dict of DataFrames: main, dm_tests, mcs, subsample.
    """
    schemes = schemes or OOS_SCHEMES
    horizons = horizons or OOS_HORIZONS

    # Load all forecast files
    fc = {}
    for m in model_names:
        for s in schemes:
            for h in horizons:
                p = OOS_DIR / f"{m}_{s}_h{h}.csv"
                if p.exists():
                    fc[(m, s, h)] = pd.read_csv(p, parse_dates=["origin_date", "forecast_target_date"])

    # ── Main QLIKE / RMSE table ────────────────────────────────────────────
    main_rows = []
    for m in model_names:
        for s in schemes:
            for h in horizons:
                key = (m, s, h)
                if key not in fc:
                    continue
                d = fc[key].dropna(subset=["forecast_variance", "realized_variance"])
                main_rows.append({
                    "model":   m, "scheme": s, "horizon": h,
                    "n_obs":   len(d),
                    "QLIKE":   qlike(d["realized_variance"], d["forecast_variance"]),
                    "RMSE":    rmse (d["realized_variance"], d["forecast_variance"]),
                })
    main_df = pd.DataFrame(main_rows)

    # ── DM tests (vs each baseline, on QLIKE) ─────────────────────────────
    dm_rows = []
    for baseline in baselines:
        for m in model_names:
            if m == baseline:
                continue
            for s in schemes:
                for h in horizons:
                    if (m, s, h) not in fc or (baseline, s, h) not in fc:
                        continue
                    a = fc[(m, s, h)].set_index("forecast_target_date")
                    b = fc[(baseline, s, h)].set_index("forecast_target_date")
                    common = a.index.intersection(b.index)
                    if len(common) < 30:
                        continue
                    a, b = a.loc[common], b.loc[common]
                    eps = 1e-12
                    la = np.log(a["forecast_variance"] + eps) + a["realized_variance"] / (a["forecast_variance"] + eps)
                    lb = np.log(b["forecast_variance"] + eps) + b["realized_variance"] / (b["forecast_variance"] + eps)
                    md, ds, p = dm_test(la, lb, horizon=h)
                    dm_rows.append({
                        "model":         m,
                        "vs_baseline":   baseline,
                        "scheme":        s, "horizon": h,
                        "mean_diff_QL":  md,
                        "DM_stat":       ds,
                        "p_value":       p,
                        "interp":        ("model better" if (p < 0.05 and md < 0)
                                          else "baseline better" if (p < 0.05 and md > 0)
                                          else "no diff."),
                    })
    dm_df = pd.DataFrame(dm_rows)

    # ── MCS at 90% (per scheme, horizon) ──────────────────────────────────
    mcs_rows = []
    for s in schemes:
        for h in horizons:
            losses = {}
            common = None
            for m in model_names:
                if (m, s, h) not in fc:
                    continue
                d = fc[(m, s, h)].set_index("forecast_target_date").dropna(
                    subset=["forecast_variance", "realized_variance"])
                eps = 1e-12
                lvec = np.log(d["forecast_variance"] + eps) + d["realized_variance"] / (d["forecast_variance"] + eps)
                losses[m] = lvec
                common = lvec.index if common is None else common.intersection(lvec.index)
            if not losses or common is None or len(common) < 30:
                continue
            losses_aligned = {k: v.loc[common].values for k, v in losses.items()}
            in_set = model_confidence_set(losses_aligned, alpha=0.10)
            mcs_rows.append({
                "scheme": s, "horizon": h,
                "n_obs":  len(common),
                "in_MCS": ", ".join(in_set),
                "excl":   ", ".join([m for m in losses_aligned if m not in in_set]),
            })
    mcs_df = pd.DataFrame(mcs_rows)

    # ── Sub-sample QLIKE per (model, scheme, horizon, window) ─────────────
    sub_rows = []
    for label, (start, end) in subsamples.items():
        start_ts = pd.Timestamp(start); end_ts = pd.Timestamp(end)
        for m in model_names:
            for s in schemes:
                for h in horizons:
                    key = (m, s, h)
                    if key not in fc:
                        continue
                    d = fc[key]
                    sub = d[(d["forecast_target_date"] >= start_ts) &
                            (d["forecast_target_date"] <= end_ts)].dropna(
                        subset=["forecast_variance", "realized_variance"])
                    if len(sub) < 5:
                        continue
                    sub_rows.append({
                        "subsample": label,
                        "model":     m, "scheme": s, "horizon": h,
                        "n_obs":     len(sub),
                        "QLIKE":     qlike(sub["realized_variance"], sub["forecast_variance"]),
                        "RMSE":      rmse (sub["realized_variance"], sub["forecast_variance"]),
                    })
    sub_df = pd.DataFrame(sub_rows)

    # ── B5.1 abs_return-target re-eval (re-uses B2.2 forecasts) ─────────────
    # Compare on volatility scale (sqrt forecast variance vs |return| target)
    # so units match. The original variance-vs-|return| comparison gave QLIKE
    # values ~70 due to the unit mismatch (Patton 2011, Bollerslev et al. 2016).
    abs_rows = []
    for m in ("B2.2",):
        for s in schemes:
            for h in horizons:
                key = (m, s, h)
                if key not in fc:
                    continue
                d = fc[key].dropna(subset=["forecast_variance", "abs_return_target"])
                forecast_vol = np.sqrt(np.maximum(d["forecast_variance"].values, 0.0))
                abs_rows.append({
                    "robustness": "B5.1 abs_return target, vol scale (reuses B2.2 forecasts)",
                    "scheme": s, "horizon": h, "n_obs": len(d),
                    "QLIKE":  qlike(d["abs_return_target"].values, forecast_vol),
                    "RMSE":   rmse (d["abs_return_target"].values, forecast_vol),
                })
    abs_df = pd.DataFrame(abs_rows)

    return {"main": main_df, "dm": dm_df, "mcs": mcs_df,
            "subsample": sub_df, "abs_return": abs_df}


# ─────────────────────────────────────────────────────────────────────────────
# 6. FORECAST COMBINATION (B1.2 + B2.2)
# ─────────────────────────────────────────────────────────────────────────────

def forecast_combination(model_a="B1.2", model_b="B2.2",
                         schemes=("RW", "IW"), horizons=(1, 5),
                         w_grid=None):
    if w_grid is None:
        w_grid = np.linspace(0.0, 1.0, 21)
    rows = []
    for s in schemes:
        for h in horizons:
            pa = OOS_DIR / f"{model_a}_{s}_h{h}.csv"
            pb = OOS_DIR / f"{model_b}_{s}_h{h}.csv"
            if not (pa.exists() and pb.exists()):
                continue
            a = pd.read_csv(pa, parse_dates=["forecast_target_date"]).set_index("forecast_target_date")
            b = pd.read_csv(pb, parse_dates=["forecast_target_date"]).set_index("forecast_target_date")
            common = a.index.intersection(b.index)
            if len(common) < 30:
                continue
            a, b = a.loc[common], b.loc[common]
            best_w, best_q = 0.0, np.inf
            for w in w_grid:
                fc = (1 - w) * a["forecast_variance"].values + w * b["forecast_variance"].values
                q = qlike(a["realized_variance"].values, fc)
                if q < best_q:
                    best_q = q; best_w = float(w)
            q_a = qlike(a["realized_variance"], a["forecast_variance"])
            rows.append({
                "scheme": s, "horizon": h,
                "model_a": model_a, "model_b": model_b,
                "n_obs":  len(common),
                "QLIKE_a_alone":  q_a,
                "best_w":         best_w,
                "QLIKE_combined": best_q,
                "Δ_QLIKE":        best_q - q_a,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ECONOMIC SIGNIFICANCE OF NEWS EFFECT (B2.2)
# ─────────────────────────────────────────────────────────────────────────────

def economic_significance(b22_fit):
    """
    Implied range of the f(P, N) scaling factor under fitted B2.2 parameters.
    """
    p = b22_fit.params
    a = p["a"]; b = p["b"]; kappa = p["kappa"]; gamma = p["gamma"]

    def f(P, N):
        return a + 0.5 * b * (np.tanh(0.5 * kappa * P) - np.tanh(0.5 * gamma * N))

    return {
        "f(P=0, N=0)":          f(0.0, 0.0),
        "f(P=1, N=0) max-dov":  f(1.0, 0.0),
        "f(P=0, N=-1) max-haw": f(0.0, -1.0),
        "f(P=1, N=-1) max-mix": f(1.0, -1.0),
        "max":                  max(f(0,0), f(1,0), f(0,-1), f(1,-1)),
        "min":                  min(f(0,0), f(1,0), f(0,-1), f(1,-1)),
        "range_pct_baseline":   (max(f(0,0), f(1,0), f(0,-1), f(1,-1)) -
                                  min(f(0,0), f(1,0), f(0,-1), f(1,-1))) / a * 100.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. UNIFIED REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(args, in_main, in_robust, lb_diag, fits_main,
                  oos_eval, fc_combo, econ_sig, in_appendix=None):
    out = []
    push = out.append
    push("=" * 78)
    push("FINAL RESULTS — VOLATILITY FORECASTING WITH ECB NEWS SENTIMENT")
    push("=" * 78)
    push("")
    push("Pipeline configuration")
    push("-" * 78)
    push(f"  In-sample window:  2008-01-01 → 2018-12-31")
    push(f"  Out-of-sample:     2019-01-01 → 2023-12-31")
    push(f"  Innovations:       Student-t (default)")
    push(f"  Vol proxy (main):  squared log return")
    push(f"  News scoring:      FOMC-RoBERTa stance, sentence-level, τ=0.50, "
         f"persist-to-next-event smoothing, Sadik in-sample-max rescaling")
    push("")

    # ── In-sample table (main) ─────────────────────────────────────────────
    push("Table 1A — In-sample MLE, main suite")
    push("-" * 78)
    push(in_main.round(3).to_string(index=False))
    push("")

    # ── In-sample table (robustness) ───────────────────────────────────────
    if in_robust is not None and len(in_robust):
        push("Table 1B — In-sample MLE, robustness suite")
        push("-" * 78)
        push(in_robust.round(3).to_string(index=False))
        push("")

    # ── Residual diagnostics ───────────────────────────────────────────────
    if lb_diag is not None and len(lb_diag):
        push("Table 1C — Ljung-Box residual diagnostics (main suite)")
        push("  H0 of no autocorrelation; small p indicates remaining structure.")
        push("-" * 78)
        push(lb_diag.round(3).to_string(index=False))
        push("")

    # ── OOS Main ───────────────────────────────────────────────────────────
    if oos_eval is not None and len(oos_eval.get("main", [])):
        push("Table 2 — OOS QLIKE / RMSE")
        push("-" * 78)
        push(oos_eval["main"].round(6).to_string(index=False))
        push("")
    if oos_eval is not None and len(oos_eval.get("dm", [])):
        push("Table 3 — Diebold-Mariano tests (vs B1.1 GARCH and vs B1.2 GJR)")
        push("  mean_diff_QL = QLIKE(model) − QLIKE(baseline); negative = model better.")
        push("-" * 78)
        push(oos_eval["dm"].round(4).to_string(index=False))
        push("")
    if oos_eval is not None and len(oos_eval.get("mcs", [])):
        push("Table 4 — Model Confidence Set at 90%")
        push("-" * 78)
        push(oos_eval["mcs"].to_string(index=False))
        push("")
    if oos_eval is not None and len(oos_eval.get("subsample", [])):
        push("Table 5 — Sub-sample OOS analysis")
        push("  Sub-windows: 2019_normal, 2020_2021_covid, 2022_2023_infl")
        push("-" * 78)
        push(oos_eval["subsample"].round(6).to_string(index=False))
        push("")
    if oos_eval is not None and len(oos_eval.get("abs_return", [])):
        push("Table 6 — Robustness: |return| target (B5.1, reuses B2.2 forecasts)")
        push("-" * 78)
        push(oos_eval["abs_return"].round(6).to_string(index=False))
        push("")

    # ── Forecast combination ───────────────────────────────────────────────
    if fc_combo is not None and len(fc_combo):
        push("Table 7 — Forecast combination: (1−w) · B1.2 + w · B2.2")
        push("  best_w near 0 ⇒ B2.2 carries no orthogonal information vs GJR.")
        push("-" * 78)
        push(fc_combo.round(6).to_string(index=False))
        push("")

    # ── Economic significance ──────────────────────────────────────────────
    if econ_sig is not None:
        push("Table 8 — Economic significance of news scaling under B2.2")
        push("-" * 78)
        for k, v in econ_sig.items():
            push(f"  {k:<24s} = {v:+.6f}" if not k.startswith("range") else
                 f"  {k:<24s} = {v:+.2f}%")
        push("")

    # ── Appendix ───────────────────────────────────────────────────────────
    if in_appendix is not None and len(in_appendix):
        push("Appendix — Surprise specifications (in-sample only)")
        push("  All three underperform raw levels (B2.2). Reported for completeness.")
        push("-" * 78)
        push(in_appendix.round(3).to_string(index=False))
        push("")

    push("=" * 78)
    push("END")
    push("=" * 78)

    text = "\n".join(out) + "\n"
    (OUTPUT_DIR / "final_results_summary.txt").write_text(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-sample-only", action="store_true")
    parser.add_argument("--skip-oos", action="store_true",
                        help="In-sample + diagnostics only; reuse existing OOS forecast files if present.")
    parser.add_argument("--main-only", action="store_true",
                        help="Skip robustness rows.")
    parser.add_argument("--include-appendix", action="store_true",
                        help="Also fit B2.3 (MPS-decayed surprises) in-sample.")
    parser.add_argument("--quick-oos", action="store_true",
                        help="Run OOS with only 20 origins (smoke test).")
    parser.add_argument("--gaussian", action="store_true",
                        help="Use Gaussian innovations (default: Student-t).")
    args = parser.parse_args()

    dist = "normal" if args.gaussian else "studentst"
    max_origins = 20 if args.quick_oos else None

    log.info("=" * 78)
    log.info("RUN_MAIN_PIPELINE — final modeling orchestrator")
    log.info("=" * 78)

    # ── 1. Load + augment ─────────────────────────────────────────────────
    log.info("\n[1] Load + augment master dataset")
    df = load_master()
    df = augment_thresholds(df)
    df = augment_bulletins(df)

    # Determine which robustness models we can actually run
    is_only_models = []
    if not args.main_only:
        if {"P_t_bulletins", "N_t_bulletins"}.issubset(df.columns):
            is_only_models.append("B5.5")
        if {"P_t_t000", "N_t_t000"}.issubset(df.columns):
            is_only_models.append("B5.6")
        if {"P_t_t080", "N_t_t080"}.issubset(df.columns):
            is_only_models.append("B5.7")
    oos_robust = []
    if not args.main_only:
        oos_robust = ["B5.3", "B5.4"] + (["B5.5"] if "B5.5" in is_only_models else [])

    # ── 2. In-sample suite (main) ────────────────────────────────────────
    log.info("\n[2] In-sample MLE — main suite")
    in_main, fits_main = fit_in_sample_suite(df, MAIN_MODELS, dist=dist)
    in_main.to_csv(OUTPUT_DIR / "final_in_sample_main.csv", index=False)

    # ── 3. In-sample suite (robustness) ──────────────────────────────────
    in_robust = None
    if not args.main_only:
        log.info("\n[3] In-sample MLE — robustness suite")
        rob_models = ["B5.3", "B5.4"] + is_only_models
        in_robust, _ = fit_in_sample_suite(df, rob_models, dist=dist)

        # B5.8 — Gaussian innovations on B2.2 (in-sample only)
        log.info("\n  B5.8 — B2.2 with Gaussian innovations")
        try:
            m, ins = fit_in_sample(df, "B2.2", dist="normal")
            in_robust = pd.concat([in_robust, pd.DataFrame([{
                "model": "B5.8 (B2.2 Gaussian)", "dist": "normal",
                "n_obs": len(ins), "loglik": m.loglik, "aic": m.aic,
                "n_params": len(getattr(m, "_theta_opt", m.params)), "fit_time": np.nan,
            }])], ignore_index=True)
            log.info(f"    B5.8: ll={m.loglik:.2f}  AIC={m.aic:.2f}")
        except Exception as e:
            log.warning(f"    B5.8 failed: {e}")
        in_robust.to_csv(OUTPUT_DIR / "final_in_sample_robust.csv", index=False)

    # ── 4. Residual diagnostics ──────────────────────────────────────────
    log.info("\n[4] Ljung-Box residual diagnostics (main suite)")
    lb_diag = residual_diagnostics(fits_main, lags=LJUNG_BOX_LAGS)
    lb_diag.to_csv(OUTPUT_DIR / "final_residual_diagnostics.csv", index=False)
    log.info(f"  → final_residual_diagnostics.csv ({len(lb_diag)} rows)")

    # ── 5. OOS forecasting (gated) ───────────────────────────────────────
    if not (args.in_sample_only or args.skip_oos):
        log.info("\n[5] OOS forecasting — main suite")
        main_specs = [(m, s, h) for m in MAIN_MODELS for s in OOS_SCHEMES for h in OOS_HORIZONS]
        run_oos_suite(df, main_specs, dist=dist, max_origins=max_origins)
        if oos_robust:
            log.info("\n[5b] OOS forecasting — robustness suite (RW only)")
            rob_specs = [(m, "RW", h) for m in oos_robust for h in OOS_HORIZONS]
            run_oos_suite(df, rob_specs, dist=dist, max_origins=max_origins)
    else:
        log.info("\n[5] OOS forecasting — SKIPPED (will reuse existing oos_forecasts/ files for evaluation)")

    # ── 6. OOS evaluation ────────────────────────────────────────────────
    oos_eval = None
    if not args.in_sample_only:
        log.info("\n[6] OOS evaluation")
        all_models_for_eval = MAIN_MODELS + oos_robust
        oos_eval = evaluate_oos(all_models_for_eval, baselines=("B1.1", "B1.2"))
        oos_eval["main"].to_csv(OUTPUT_DIR / "final_oos_main.csv", index=False)
        oos_eval["dm"].to_csv(OUTPUT_DIR / "final_oos_dm.csv", index=False)
        oos_eval["mcs"].to_csv(OUTPUT_DIR / "final_oos_mcs.csv", index=False)
        oos_eval["subsample"].to_csv(OUTPUT_DIR / "final_oos_subsample.csv", index=False)
        oos_eval["abs_return"].to_csv(OUTPUT_DIR / "final_oos_absreturn.csv", index=False)
        log.info(f"  → final_oos_*.csv ({sum(len(v) for v in oos_eval.values())} total rows)")

    # ── 7. Forecast combination ──────────────────────────────────────────
    fc_combo = None
    if not args.in_sample_only:
        log.info("\n[7] Forecast combination test (B1.2 + B2.2)")
        fc_combo = forecast_combination()
        if len(fc_combo):
            fc_combo.to_csv(OUTPUT_DIR / "final_forecast_combination.csv", index=False)
            log.info(f"  → final_forecast_combination.csv ({len(fc_combo)} rows)")

    # ── 8. Economic significance (B2.2) ──────────────────────────────────
    log.info("\n[8] Economic significance (B2.2)")
    econ_sig = None
    if "B2.2" in fits_main:
        m22, _ = fits_main["B2.2"]
        econ_sig = economic_significance(m22)
        for k, v in econ_sig.items():
            log.info(f"    {k:<28s} = {v:+.4f}")

    # ── 9. Appendix surprise specs ───────────────────────────────────────
    in_appendix = None
    if args.include_appendix:
        log.info("\n[9] Appendix — surprise specifications")
        in_appendix, _ = fit_in_sample_suite(df, ["B2.3"], dist=dist)
        in_appendix.to_csv(OUTPUT_DIR / "final_in_sample_appendix.csv", index=False)

    # ── 10. Unified report ───────────────────────────────────────────────
    log.info("\n[10] Writing unified report")
    text = write_summary(args, in_main, in_robust, lb_diag, fits_main,
                         oos_eval, fc_combo, econ_sig, in_appendix)
    log.info(f"  → {OUTPUT_DIR / 'final_results_summary.txt'}")
    log.info("\n" + "=" * 78)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
