"""
evaluate.py — OOS evaluation: QLIKE / RMSE / DM tests / MCS / figures
======================================================================

Consumes the OOS forecasts saved by forecast_oos.py and produces all the
empirical results for the paper:

    Tables (CSVs ready for LaTeX-formatting):
        oos_evaluation_main.csv       — Table 2: QLIKE/RMSE for the 5 main
                                                  models × 4 schemes
        oos_dm_tests.csv              — Table 3: Diebold-Mariano vs B1.1
                                                  (with HAC SE for h=5)
        oos_mcs.csv                   — Table 5: Model Confidence Set
                                                  (Hansen-Lunde-Nason 2011)
        oos_robustness.csv            — Table 4: B5.1 abs_return proxy +
                                                  B5.3 MPS-only + B5.4 speech-only

    Figures:
        figure_2_insample_fit.png     — fitted σ²_t vs realized r²_t (in-sample,
                                         all 5 main models on one chart)
        figure_3_oos_forecasts.png    — OOS forecast σ²_t vs realized r²_t
                                         (RW h=1, all 5 models)

    Summary:
        oos_summary.txt               — human-readable summary

LOSS FUNCTIONS
--------------
    QLIKE   = mean[ log(σ²_t) + realized_t / σ²_t ]    (Patton 2011 form;
                                                        lower = better)
    RMSE    = sqrt(mean( (realized_t − σ²_t)² ))
    DM stat = mean(d_t) / sqrt( LRV(d_t) / T )
              where d_t = QLIKE_a − QLIKE_b
              For h > 1: Newey-West HAC long-run variance with bandwidth h−1.

MCS (Hansen, Lunde, Nason 2011)
-------------------------------
Iteratively eliminates models from the candidate set when an equivalence
test rejects them as inferior at confidence level (1 − α). We use the
T_max statistic with stationary bootstrap (arch.bootstrap.MCS).

B5.1 ALTERNATIVE VOL PROXY
--------------------------
Reuses the saved B2.2 forecasts but computes losses against `abs_return_target`
instead of `realized_variance` (sq_return). Tests sensitivity to the realized
vol proxy choice.

USAGE
-----
    python evaluate.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("./output")
OOS_DIR     = OUTPUT_DIR / "oos_forecasts"
MASTER_CSV  = OUTPUT_DIR / "model_data_master.csv"

MAIN_MODELS = ["B1.1", "B1.2", "B1.3", "B2.2", "B2.3"]
ROBUSTNESS_MODELS = ["B5.3", "B5.4"]
ALL_MODELS = MAIN_MODELS + ROBUSTNESS_MODELS

SCHEMES   = ["RW", "IW"]
HORIZONS  = [1, 5]

MCS_ALPHA = 0.10        # 90% confidence MCS
MCS_REPS  = 5000

BASELINE_MODEL = "B1.1"  # DM tests are vs this baseline


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "evaluate.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def qlike_loss(realized, forecast):
    """
    Patton (2011) robust QLIKE loss, lower = better:
        L_t = log(forecast_t) + realized_t / forecast_t

    Drops rows with non-positive forecast or non-finite realized.
    Returns: per-observation loss array (NOT mean), with NaN where invalid.
    """
    realized = np.asarray(realized, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    out = np.full_like(realized, np.nan)
    valid = (forecast > 0) & np.isfinite(realized) & np.isfinite(forecast)
    out[valid] = np.log(forecast[valid]) + realized[valid] / forecast[valid]
    return out


def squared_error_loss(realized, forecast):
    """Per-observation squared error loss for RMSE."""
    realized = np.asarray(realized, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    out = (realized - forecast) ** 2
    out[~(np.isfinite(realized) & np.isfinite(forecast))] = np.nan
    return out


def diebold_mariano(loss_a, loss_b, h=1):
    """
    Diebold-Mariano test of equal predictive accuracy.

    Two-sided p-value. For h > 1, long-run variance estimated via
    Newey-West with bandwidth = h − 1 (standard for h-step forecasts
    where loss differentials inherit at most h-1 lags of correlation).

    Returns: (DM_stat, p_value, mean_diff).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    valid = np.isfinite(d)
    d = d[valid]
    T = len(d)
    if T < 10:
        return float("nan"), float("nan"), float("nan")

    d_bar = float(np.mean(d))

    if h <= 1:
        # i.i.d. asymptotic: V = sample variance
        var_d = float(np.var(d, ddof=1))
    else:
        # Newey-West HAC with bandwidth h-1, Bartlett kernel
        gamma_0 = float(np.var(d, ddof=0))
        var_d = gamma_0
        for k in range(1, h):
            cov_k = float(np.cov(d[:-k], d[k:], ddof=0)[0, 1])
            var_d += 2.0 * (1.0 - k / h) * cov_k
        var_d = max(var_d, 1e-12)

    se = np.sqrt(var_d / T)
    stat = d_bar / max(se, 1e-12)
    p_value = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return float(stat), float(p_value), d_bar


# ─────────────────────────────────────────────────────────────────────────────
# 2. MCS (via arch.bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def model_confidence_set(losses_df, alpha=MCS_ALPHA, reps=MCS_REPS, seed=42):
    """
    Hansen-Lunde-Nason (2011) Model Confidence Set.

    losses_df: DataFrame with one column per model, T rows of losses.

    Returns: dict per model with {'in_mcs': bool, 'p_value': float, 'rank': int}.
    Models in the MCS are those that cannot be statistically eliminated as
    inferior at confidence level (1 − alpha).
    """
    try:
        from arch.bootstrap import MCS
    except ImportError:
        log.warning("arch.bootstrap.MCS not available — skipping MCS")
        return {}

    losses_df = losses_df.dropna(how="any")
    if losses_df.shape[0] < 30 or losses_df.shape[1] < 2:
        log.warning(f"  MCS skipped: insufficient data "
                    f"({losses_df.shape[0]} rows, {losses_df.shape[1]} models)")
        return {}

    mcs = MCS(losses_df.values, size=alpha, reps=reps, seed=seed,
              method="max")  # T_max statistic
    mcs.compute()
    pvals = pd.Series(mcs.pvalues.iloc[:, 0].values,
                      index=losses_df.columns).sort_values()
    included = set(losses_df.columns[mcs.included])

    out = {}
    rank = 1
    for model, p in pvals.items():
        out[model] = {
            "p_value": float(p),
            "in_mcs": bool(model in included),
            "rank":   rank,
        }
        rank += 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD FORECASTS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_forecasts():
    """Load all forecast CSVs into one big DataFrame."""
    rows = []
    for f in sorted(OOS_DIR.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["origin_date", "forecast_target_date"])
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No forecasts found in {OOS_DIR}")
    out = pd.concat(rows, ignore_index=True)
    log.info(f"  Loaded {len(out):,} forecasts across "
             f"{out['model'].nunique()} models, "
             f"{out.groupby(['model','scheme','horizon']).ngroups} (model,scheme,horizon) combos.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. TABLE 2 — Main OOS evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_main_table(fc):
    """
    Table 2: QLIKE and RMSE per (model, scheme, horizon).
    """
    rows = []
    for (model, scheme, h), g in fc.groupby(["model", "scheme", "horizon"]):
        if model not in MAIN_MODELS:
            continue
        ql = qlike_loss(g["realized_variance"], g["forecast_variance"])
        sq = squared_error_loss(g["realized_variance"], g["forecast_variance"])
        rows.append({
            "model": model, "scheme": scheme, "horizon": int(h),
            "n_obs":  int(np.sum(np.isfinite(ql))),
            "QLIKE":  float(np.nanmean(ql)),
            "RMSE":   float(np.sqrt(np.nanmean(sq))),
        })
    return pd.DataFrame(rows).sort_values(["scheme", "horizon", "QLIKE"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. TABLE 3 — Diebold-Mariano tests vs B1.1
# ─────────────────────────────────────────────────────────────────────────────

def build_dm_table(fc, baseline=BASELINE_MODEL):
    """DM tests of each main model's QLIKE vs B1.1's QLIKE, by scheme/horizon."""
    rows = []
    for scheme in SCHEMES:
        for h in HORIZONS:
            base = fc[(fc["model"] == baseline)
                      & (fc["scheme"] == scheme)
                      & (fc["horizon"] == h)].sort_values("origin_date")
            if base.empty:
                continue
            base_loss = qlike_loss(base["realized_variance"], base["forecast_variance"])
            for model in MAIN_MODELS:
                if model == baseline:
                    rows.append({
                        "model": model, "scheme": scheme, "horizon": int(h),
                        "DM_stat": float("nan"), "p_value": float("nan"),
                        "mean_diff_QLIKE": 0.0,
                        "interpretation": "(baseline)",
                    })
                    continue
                sub = fc[(fc["model"] == model)
                         & (fc["scheme"] == scheme)
                         & (fc["horizon"] == h)].sort_values("origin_date")
                if sub.empty or len(sub) != len(base):
                    continue
                cand_loss = qlike_loss(sub["realized_variance"], sub["forecast_variance"])
                stat, p, mean_diff = diebold_mariano(cand_loss, base_loss, h=int(h))
                # Interpret: mean_diff > 0 means model has HIGHER loss (worse) than baseline
                if not np.isfinite(p):
                    interp = "n/a"
                elif p < 0.05:
                    interp = ("model worse" if mean_diff > 0 else "model better")
                else:
                    interp = "no diff."
                rows.append({
                    "model": model, "scheme": scheme, "horizon": int(h),
                    "DM_stat": stat, "p_value": p,
                    "mean_diff_QLIKE": mean_diff,
                    "interpretation": interp,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TABLE 5 — Model Confidence Set
# ─────────────────────────────────────────────────────────────────────────────

def build_mcs_table(fc):
    """Run MCS within each (scheme, horizon) over the main models' QLIKE losses."""
    rows = []
    for scheme in SCHEMES:
        for h in HORIZONS:
            losses_per_model = {}
            for model in MAIN_MODELS:
                sub = fc[(fc["model"] == model)
                         & (fc["scheme"] == scheme)
                         & (fc["horizon"] == h)].sort_values("origin_date")
                if sub.empty:
                    continue
                losses_per_model[model] = qlike_loss(
                    sub["realized_variance"].values, sub["forecast_variance"].values
                )
            if len(losses_per_model) < 2:
                continue
            losses_df = pd.DataFrame(losses_per_model)
            mcs_result = model_confidence_set(losses_df, alpha=MCS_ALPHA, reps=MCS_REPS)
            for m, info in mcs_result.items():
                rows.append({
                    "scheme": scheme, "horizon": int(h), "model": m,
                    "in_mcs": info["in_mcs"], "p_value": info["p_value"],
                    "rank": info["rank"],
                })
    return pd.DataFrame(rows).sort_values(["scheme", "horizon", "rank"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. TABLE 4 — Robustness
# ─────────────────────────────────────────────────────────────────────────────

def build_robustness_table(fc):
    """
    Robustness columns:
      B5.1 — alternative vol proxy: same B2.2 forecasts evaluated against
             abs_return_target (instead of sq_return)
      B5.3 — MPS-only stance variant
      B5.4 — Speeches-only stance variant

    For each, RW only, h=1 and h=5. QLIKE only (RMSE for completeness).
    """
    rows = []

    # B5.1 — recompute B2.2 losses against abs_return_target.
    # Units fix: abs_return_target is in |return| units while forecast_variance
    # is in squared-return units. Compare on the volatility scale by taking
    # sqrt(forecast_variance) so loss is computed apples-to-apples. Standard
    # in vol-forecasting literature when |r| is the proxy (e.g. Bollerslev et al.
    # 2016, Patton 2011).
    for h in HORIZONS:
        sub = fc[(fc["model"] == "B2.2") & (fc["scheme"] == "RW")
                 & (fc["horizon"] == h)]
        if sub.empty:
            continue
        forecast_vol = np.sqrt(np.maximum(sub["forecast_variance"].values, 0.0))
        ql_alt = qlike_loss(sub["abs_return_target"].values, forecast_vol)
        sq_alt = squared_error_loss(sub["abs_return_target"].values, forecast_vol)
        rows.append({
            "robustness": "B5.1 abs_return proxy (re-evaluated B2.2, vol-scale)",
            "scheme": "RW", "horizon": int(h),
            "n_obs": int(np.sum(np.isfinite(ql_alt))),
            "QLIKE": float(np.nanmean(ql_alt)),
            "RMSE":  float(np.sqrt(np.nanmean(sq_alt))),
        })

    # B5.3, B5.4 — separate models
    for model in ROBUSTNESS_MODELS:
        for h in HORIZONS:
            sub = fc[(fc["model"] == model) & (fc["scheme"] == "RW")
                     & (fc["horizon"] == h)]
            if sub.empty:
                continue
            ql = qlike_loss(sub["realized_variance"], sub["forecast_variance"])
            sq = squared_error_loss(sub["realized_variance"], sub["forecast_variance"])
            rows.append({
                "robustness": f"{model} {('MPS-only' if model=='B5.3' else 'Speeches-only')}",
                "scheme": "RW", "horizon": int(h),
                "n_obs": int(np.sum(np.isfinite(ql))),
                "QLIKE": float(np.nanmean(ql)),
                "RMSE":  float(np.sqrt(np.nanmean(sq))),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 2 — In-sample fitted vs realized
# ─────────────────────────────────────────────────────────────────────────────

def make_figure_2(out_path):
    """
    Re-fit each main model on the in-sample period and plot
    fitted σ²_t vs realized r²_t over time.
    """
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv(MASTER_CSV, parse_dates=["date"])
    ins = df[df["period"] == "insample"].dropna(
        subset=["log_return", "P_t", "N_t", "P_mps_surprise", "N_mps_surprise"]
    ).reset_index(drop=True)

    from models import GARCH11, GJRGARCH, EGARCH, NAGarchAsym

    log.info("  Re-fitting main models for Figure 2...")
    fits = {}
    fits["B1.1 GARCH"]    = GARCH11(ins["log_return"]).fit().conditional_variance()
    fits["B1.2 GJR"]      = GJRGARCH(ins["log_return"]).fit().conditional_variance()
    fits["B1.3 EGARCH"]   = EGARCH(ins["log_return"]).fit().conditional_variance()
    fits["B2.2 levels"]   = NAGarchAsym(ins["log_return"], P=ins["P_t"], N=ins["N_t"]
                                       ).fit(n_restarts=2).conditional_variance()
    fits["B2.3 surprise"] = NAGarchAsym(ins["log_return"],
                                        P=ins["P_mps_surprise"],
                                        N=ins["N_mps_surprise"]
                                       ).fit(n_restarts=2).conditional_variance()

    fig, ax = plt.subplots(figsize=(14, 5))
    dates = pd.to_datetime(ins["date"]).values
    ax.plot(dates, ins["sq_return"].values, lw=0.4, color="grey", alpha=0.5,
            label="realized r²_t")
    palette = {
        "B1.1 GARCH":    "tab:blue",
        "B1.2 GJR":      "tab:purple",
        "B1.3 EGARCH":   "tab:green",
        "B2.2 levels":   "tab:orange",
        "B2.3 surprise": "tab:red",
    }
    for name, sig2 in fits.items():
        # `conditional_variance()` returns a pandas Series in raw units; just plot
        ax.plot(dates, np.asarray(sig2), lw=0.7, color=palette[name],
                alpha=0.85, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("σ²_t (log scale)")
    ax.set_xlabel("date")
    ax.set_title("Figure 2: In-sample fitted σ²_t for all main models, vs realized r²_t")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURE 3 — OOS forecasts vs realized
# ─────────────────────────────────────────────────────────────────────────────

def make_figure_3(fc, out_path):
    """OOS h=1 RW forecasts: each model's σ²_t vs realized r²_t."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Realized: pull from any model's data (they're all aligned)
    ref = fc[(fc["model"] == "B1.1") & (fc["scheme"] == "RW")
             & (fc["horizon"] == 1)].sort_values("forecast_target_date")
    ax.plot(ref["forecast_target_date"], ref["realized_variance"],
            lw=0.4, color="grey", alpha=0.5, label="realized r²_t")

    palette = {
        "B1.1": "tab:blue", "B1.2": "tab:purple", "B1.3": "tab:green",
        "B2.2": "tab:orange", "B2.3": "tab:red",
    }
    for model in MAIN_MODELS:
        sub = fc[(fc["model"] == model) & (fc["scheme"] == "RW")
                 & (fc["horizon"] == 1)].sort_values("forecast_target_date")
        if sub.empty:
            continue
        ax.plot(sub["forecast_target_date"], sub["forecast_variance"],
                lw=0.7, color=palette[model], alpha=0.85, label=model)
    ax.set_yscale("log")
    ax.set_ylabel("σ²_t / r²_t (log scale)")
    ax.set_xlabel("date")
    ax.set_title("Figure 3: OOS h=1 forecasts (RW) vs realized r²_t, 2019–2023")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(main_tbl, dm_tbl, mcs_tbl, robust_tbl, out_path):
    lines = []
    lines.append("OOS EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Table 2 — Main OOS evaluation (QLIKE, RMSE):")
    lines.append(main_tbl.round(6).to_string(index=False))
    lines.append("")
    lines.append("Table 3 — Diebold-Mariano vs B1.1 baseline (QLIKE):")
    dm_pretty = dm_tbl[["model","scheme","horizon","mean_diff_QLIKE","DM_stat","p_value","interpretation"]]
    lines.append(dm_pretty.round(4).to_string(index=False))
    lines.append("")
    lines.append("Table 5 — Model Confidence Set (90%):")
    if not mcs_tbl.empty:
        for (scheme, h), g in mcs_tbl.groupby(["scheme", "horizon"]):
            lines.append(f"  scheme={scheme}, h={h}:")
            in_mcs = g[g["in_mcs"]]["model"].tolist()
            out_mcs = g[~g["in_mcs"]]["model"].tolist()
            lines.append(f"    in MCS  : {in_mcs}")
            lines.append(f"    excluded: {out_mcs}")
    lines.append("")
    lines.append("Table 4 — Robustness (RW only):")
    lines.append(robust_tbl.round(6).to_string(index=False))
    out_path.write_text("\n".join(lines))
    log.info(f"  → {out_path}")
    log.info("\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("EVALUATE — OOS losses, DM tests, MCS, robustness, figures")
    log.info("=" * 70)

    log.info("\n── Loading forecasts ──")
    fc = load_all_forecasts()

    # Drop rows with NaN realized_variance (h=5 boundary effect at the end of OOS)
    n_before = len(fc)
    fc = fc.dropna(subset=["realized_variance"])
    log.info(f"  Dropped {n_before - len(fc)} rows with NaN realized_variance "
             f"(typical h=5 right-edge effect).")

    log.info("\n── Table 2: Main OOS evaluation ──")
    main_tbl = build_main_table(fc)
    main_tbl.to_csv(OUTPUT_DIR / "oos_evaluation_main.csv", index=False)
    log.info(f"  → {OUTPUT_DIR / 'oos_evaluation_main.csv'}")
    log.info("\n" + main_tbl.round(6).to_string(index=False))

    log.info("\n── Table 3: Diebold-Mariano tests vs B1.1 ──")
    dm_tbl = build_dm_table(fc)
    dm_tbl.to_csv(OUTPUT_DIR / "oos_dm_tests.csv", index=False)
    log.info(f"  → {OUTPUT_DIR / 'oos_dm_tests.csv'}")
    log.info("\n" + dm_tbl.round(4).to_string(index=False))

    log.info("\n── Table 5: Model Confidence Set (Hansen-Lunde-Nason) ──")
    mcs_tbl = build_mcs_table(fc)
    if not mcs_tbl.empty:
        mcs_tbl.to_csv(OUTPUT_DIR / "oos_mcs.csv", index=False)
        log.info(f"  → {OUTPUT_DIR / 'oos_mcs.csv'}")
        log.info("\n" + mcs_tbl.to_string(index=False))

    log.info("\n── Table 4: Robustness ──")
    robust_tbl = build_robustness_table(fc)
    robust_tbl.to_csv(OUTPUT_DIR / "oos_robustness.csv", index=False)
    log.info(f"  → {OUTPUT_DIR / 'oos_robustness.csv'}")
    log.info("\n" + robust_tbl.round(6).to_string(index=False))

    log.info("\n── Figure 2: In-sample fitted vs realized ──")
    make_figure_2(OUTPUT_DIR / "figure_2_insample_fit.png")

    log.info("\n── Figure 3: OOS forecasts vs realized ──")
    make_figure_3(fc, OUTPUT_DIR / "figure_3_oos_forecasts.png")

    log.info("\n── Summary ──")
    write_summary(main_tbl, dm_tbl, mcs_tbl, robust_tbl,
                  OUTPUT_DIR / "oos_summary.txt")

    log.info("\n" + "=" * 70)
    log.info("EVALUATE COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
