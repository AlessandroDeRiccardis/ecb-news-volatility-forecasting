"""
estimate_in_sample.py — in-sample MLE for benchmarks + stance-encoding variants
================================================================================

Streamlined model suite (no H_t mechanism — see methodology note below):

    Benchmarks (no news):
        B1.1  GARCH(1,1)
        B1.2  GJR-GARCH(1,1)
        B1.3  EGARCH(1,1)            ← Sadik et al. (2018) primary benchmark

    Stance-encoding variants:
        B2.2  NA-GARCH-asym (levels)        — P_t, N_t (Sadik-rescaled)
        B2.3  NA-GARCH-asym (MPS surprises) — |ΔP_mps_decayed|, −|ΔN_mps_decayed|

Methodology note: an earlier draft included a binary regime indicator H_t
multiplying the news scaling (full spec in Sadik et al. 2018 augmented with
H_t × stance interaction). In-sample fits across c ∈ {1.0, ..., 2.0} found
the H_t × stance interaction adds at most +1.06 LL units over B2.2, with
the Hansen (1999) fixed-regressor bootstrap returning p = 0.46 (LR_obs =
2.13 vs bootstrap 95th percentile = 12.82) — clearly null even after
correcting for the Davies-problem inflation that chi-square asymptotics
miss. We dropped the H_t mechanism from the main spec and reframe the
contribution as a comparison of stance-signal encodings (level vs.
event-driven decayed surprise) against asymmetric-shock benchmarks.

Outputs:
    output/in_sample_estimates.csv     — Table 1 (parameters + SEs)
    output/in_sample_summary.txt       — human-readable summary

USAGE
-----
    python estimate_in_sample.py
    python estimate_in_sample.py --gaussian        # Gaussian innovations

Estimated runtime: < 1 minute.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models import (
    GARCH11, GJRGARCH, EGARCH,
    NAGarchAsym,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("./output")
MASTER_CSV  = OUTPUT_DIR / "model_data_master.csv"

DIST = "studentst"   # primary spec; pass --gaussian to override


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "estimate_in_sample.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NUMERICAL HESSIAN FOR SEs ON NA-GARCH VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

def _numerical_hessian(f, x, eps=1e-4):
    """Symmetric finite-difference Hessian of scalar f at point x."""
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        x_p = x.copy(); x_p[i] += eps
        x_m = x.copy(); x_m[i] -= eps
        H[i, i] = (f(x_p) + f(x_m) - 2.0 * f0) / (eps * eps)
    for i in range(n):
        for j in range(i + 1, n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            H[i, j] = H[j, i] = (
                f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)
            ) / (4.0 * eps * eps)
    return H


def compute_nagarch_ses(model):
    """Approximate SEs in natural-parameter space via numerical Hessian + delta method."""
    try:
        H = _numerical_hessian(model._negloglik, model._theta_opt)
        cov_uncon = np.linalg.inv(H)
        se_uncon = np.sqrt(np.maximum(np.diag(cov_uncon), 0.0))
    except Exception as e:
        log.warning(f"  Hessian inversion failed for {model.name}: {e}")
        return {k: float("nan") for k in model.params}

    theta = model._theta_opt
    se = {}
    omega = float(np.exp(theta[0]))
    se["omega"] = float(omega * se_uncon[0])
    se["alpha"] = float("nan")
    se["beta"]  = float("nan")
    news_names = model._theta_news_names()
    for i, name in enumerate(news_names):
        idx = 3 + i
        if name in ("a", "b", "kappa", "gamma"):
            nat = float(np.exp(theta[idx]))
            se[name] = float(nat * se_uncon[idx])
        elif name == "delta":
            se[name] = float(se_uncon[idx])
    if model.dist == "studentst":
        log_nu_m4 = theta[-1]
        se["nu"] = float(np.exp(log_nu_m4) * se_uncon[-1])
    return se


# ─────────────────────────────────────────────────────────────────────────────
# 2. RESULTS COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_arch_estimates(name, model, n_obs):
    """Build Table 1 long-form rows for an arch-wrapped model."""
    rows = []
    res = model._arch_res
    for param_name, est in res.params.items():
        canonical = {
            "omega":     "omega",
            "alpha[1]":  "alpha",
            "beta[1]":   "beta",
            "gamma[1]":  "gamma_lev",
            "nu":        "nu",
        }.get(param_name, param_name)
        est_use = model.params.get(canonical, float(est))
        se_raw = float(res.std_err.get(param_name, np.nan))
        if param_name == "omega" and model._vol_kind != "EGARCH":
            se_use = se_raw / (model._SCALE * model._SCALE)
        else:
            se_use = se_raw
        z = est_use / se_use if (np.isfinite(se_use) and se_use > 0) else float("nan")
        rows.append({
            "model": name, "param": canonical, "estimate": est_use,
            "se": se_use, "z": z, "loglik": model.loglik, "aic": model.aic,
            "n_obs": n_obs, "n_params": len(res.params),
        })
    return rows


def collect_nagarch_estimates(name, model, n_obs):
    """Build Table 1 long-form rows for an NA-GARCH model."""
    rows = []
    se_dict = compute_nagarch_ses(model)
    for k, v in model.params.items():
        se = se_dict.get(k, float("nan"))
        z = v / se if (np.isfinite(se) and se > 0) else float("nan")
        rows.append({
            "model": name, "param": k, "estimate": v,
            "se": se, "z": z, "loglik": model.loglik, "aic": model.aic,
            "n_obs": n_obs, "n_params": len(model._theta_opt),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian", action="store_true",
                        help="Use Gaussian innovations (default: Student-t)")
    args = parser.parse_args()

    dist = "normal" if args.gaussian else "studentst"

    log.info("=" * 70)
    log.info("ESTIMATE IN-SAMPLE — benchmarks + stance-encoding variants")
    log.info(f"  innovations: {dist}")
    log.info("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────────
    df = pd.read_csv(MASTER_CSV, parse_dates=["date"])
    ins = df[df["period"] == "insample"].dropna(
        subset=["log_return", "P_t", "N_t", "P_mps_surprise", "N_mps_surprise"]
    ).reset_index(drop=True)
    log.info(f"\n  In-sample rows: {len(ins):,} "
             f"({ins['date'].min().date()} → {ins['date'].max().date()})")

    table1_rows = []

    # ── Benchmarks B1.1, B1.2, B1.3 ─────────────────────────────────────────
    log.info("\n── Benchmarks (no news) ──")
    bench_models = {}
    for name, cls in [("B1.1 GARCH(1,1)", GARCH11),
                      ("B1.2 GJR-GARCH(1,1)", GJRGARCH),
                      ("B1.3 EGARCH(1,1)", EGARCH)]:
        m = cls(ins["log_return"], dist=dist).fit()
        bench_models[name] = m
        log.info(f"  {name:<25}  ll={m.loglik:>9,.2f}  AIC={m.aic:>10,.2f}")
        table1_rows.extend(collect_arch_estimates(name, m, n_obs=len(ins)))

    # ── B2.2 NA-GARCH-asym (levels) ─────────────────────────────────────────
    log.info("\n── B2.2 NA-GARCH-asym (LEVELS: P_t, N_t) ──")
    m_b22 = NAGarchAsym(ins["log_return"],
                        P=ins["P_t"], N=ins["N_t"],
                        dist=dist).fit(n_restarts=2)
    log.info(f"  ll={m_b22.loglik:>9,.2f}  AIC={m_b22.aic:>10,.2f}")
    log.info(f"  params={ {k: round(v,4) for k,v in m_b22.params.items()} }")
    table1_rows.extend(collect_nagarch_estimates(
        "B2.2 NA-GARCH-asym (levels)", m_b22, n_obs=len(ins)))

    # ── B2.3 NA-GARCH-asym (MPS event surprises with decay) ────────────────
    log.info("\n── B2.3 NA-GARCH-asym (MPS SURPRISES: |ΔP|, −|ΔN|, exp-decayed) ──")
    log.info(f"  Using event-driven decayed MPS surprises "
             f"(half-life 5 trading days)")
    m_b23 = NAGarchAsym(ins["log_return"],
                        P=ins["P_mps_surprise"], N=ins["N_mps_surprise"],
                        dist=dist).fit(n_restarts=2)
    log.info(f"  ll={m_b23.loglik:>9,.2f}  AIC={m_b23.aic:>10,.2f}")
    log.info(f"  params={ {k: round(v,4) for k,v in m_b23.params.items()} }")
    table1_rows.extend(collect_nagarch_estimates(
        "B2.3 NA-GARCH-asym (MPS surprises)", m_b23, n_obs=len(ins)))

    # Per-observation comparison (since rows are identical here, raw LL is fine)
    ll_diff_b23_vs_b22 = m_b23.loglik - m_b22.loglik
    log.info(f"\n  B2.3 vs B2.2: ΔLL = {ll_diff_b23_vs_b22:+.4f}")

    # ── Save Table 1 ───────────────────────────────────────────────────────
    log.info("\n── Saving Table 1 ──")
    t1 = pd.DataFrame(table1_rows)
    t1.to_csv(OUTPUT_DIR / "in_sample_estimates.csv", index=False)
    log.info(f"  → {OUTPUT_DIR / 'in_sample_estimates.csv'} "
             f"({len(t1)} rows, {t1['model'].nunique()} models)")

    # ── Summary ────────────────────────────────────────────────────────────
    summary_lines = [
        "IN-SAMPLE SUMMARY",
        "=" * 70,
        f"n_obs = {len(ins):,}",
        f"date range: {ins['date'].min().date()} → {ins['date'].max().date()}",
        f"innovations: {dist}",
        "",
        "AIC ranking (lower = better):",
    ]
    aic_summary = (
        t1.groupby("model")[["loglik", "aic", "n_params"]].first()
          .sort_values("aic")
    )
    summary_lines.append(aic_summary.round(3).to_string())
    summary_lines.append("")
    summary_lines.append(
        f"B2.3 (MPS surprise) vs B2.2 (levels): ΔLL = {ll_diff_b23_vs_b22:+.4f}, "
        f"ΔAIC = {m_b23.aic - m_b22.aic:+.4f}"
    )
    summary_lines.append(
        f"B2.2 (best news) vs B1.1 (GARCH):     ΔLL = "
        f"{m_b22.loglik - bench_models['B1.1 GARCH(1,1)'].loglik:+.4f}"
    )
    summary_lines.append(
        f"B2.2 (best news) vs B1.3 (EGARCH):    ΔLL = "
        f"{m_b22.loglik - bench_models['B1.3 EGARCH(1,1)'].loglik:+.4f}"
    )

    summary_path = OUTPUT_DIR / "in_sample_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    log.info(f"  → {summary_path}")
    log.info("\n" + "\n".join(summary_lines[-12:]))

    log.info("\n" + "=" * 70)
    log.info("ESTIMATE IN-SAMPLE COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
