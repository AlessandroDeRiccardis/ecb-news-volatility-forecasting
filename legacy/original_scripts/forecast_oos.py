"""
forecast_oos.py — out-of-sample variance forecasting
=====================================================

For each model × scheme × horizon, generates 1-step and 5-step ahead
variance forecasts at weekly intervals across the OOS period (2019-2023),
saves per-(model, scheme, horizon) CSVs that evaluate.py consumes.

MAIN SUITE — 5 models × 4 schemes (RW/IW × h=1/h=5) = 20 evaluations:
    B1.1  GARCH(1,1)
    B1.2  GJR-GARCH(1,1)
    B1.3  EGARCH(1,1)
    B2.2  NA-GARCH-asym (levels: P_t, N_t)
    B2.3  NA-GARCH-asym (MPS surprises: |ΔP_decayed|, −|ΔN_decayed|)

ROBUSTNESS — 2 models × 2 horizons (RW only) = 4 evaluations:
    B5.3  NA-GARCH-asym (MPS-only stance: P_t_mps, N_t_mps)
    B5.4  NA-GARCH-asym (speech-only stance: P_t_speech, N_t_speech)

TOTAL: 24 (model, scheme, horizon) combinations.

Plus B5.1 alternative-vol-proxy is post-hoc — it uses B2.2's saved
forecasts and just changes the loss target from sq_return to abs_return.
Computed in evaluate.py, no separate forecasting run.

OOS SCHEME DETAIL
-----------------
Forecast origin steps every WEEK_STEP trading days (default: 5).
Re-estimation cadence is one fit per origin (~260 fits per scheme).

    RW (Rolling Window): training window of fixed size = in-sample length
        (2,754 obs), slides forward dropping oldest, adding newest.
    IW (Increasing Window): training starts at the in-sample first row
        and grows to include all data through the forecast origin.

WARM-STARTING
-------------
NA-GARCH variants warm-start from the previous step's MLE. This roughly
halves estimation time per step. arch-wrapped models (B1.x) refit from
scratch each step (arch doesn't expose warm-start).

OUTPUT
------
    output/oos_forecasts/{model}_{scheme}_h{horizon}.csv
        — one row per forecast origin

USAGE
-----
    python forecast_oos.py                 # full suite
    python forecast_oos.py --models B2.2   # one model only (debug)
    python forecast_oos.py --quick         # subsample for testing

Estimated runtime (CPU, no numba): 1.5–3 hours.
With numba: ~30–60 min.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
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
OOS_DIR     = OUTPUT_DIR / "oos_forecasts"

WEEK_STEP   = 5     # trading days between successive forecast origins
DIST        = "studentst"

SCHEMES   = ["RW", "IW"]
HORIZONS  = [1, 5]

# Main models: each runs all 4 schemes (RW/IW × h=1/h=5)
MAIN_MODELS = ["B1.1", "B1.2", "B1.3", "B2.2", "B2.3"]

# Robustness models: RW only (h=1 and h=5 still tested)
ROBUSTNESS_MODELS = ["B5.3", "B5.4"]

# Schemes used per model
def schemes_for(model):
    return SCHEMES if model in MAIN_MODELS else ["RW"]


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
        logging.FileHandler(OUTPUT_DIR / "logs" / "forecast_oos.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def make_model(model_name, train_df, dist=DIST):
    """Construct an UNFITTED model instance from the training DataFrame."""
    r = train_df["log_return"]
    if model_name == "B1.1":
        return GARCH11(r, dist=dist)
    elif model_name == "B1.2":
        return GJRGARCH(r, dist=dist)
    elif model_name == "B1.3":
        return EGARCH(r, dist=dist)
    elif model_name == "B2.1":
        # NA-GARCH-net: single net stance S_t = P_t + N_t
        from models import NAGarchNet
        return NAGarchNet(r, S=train_df["S_t"], dist=dist)
    elif model_name == "B2.2":
        return NAGarchAsym(r, P=train_df["P_t"], N=train_df["N_t"], dist=dist)
    elif model_name == "B2.3":
        return NAGarchAsym(
            r, P=train_df["P_mps_surprise"], N=train_df["N_mps_surprise"], dist=dist
        )
    elif model_name == "B5.3":
        return NAGarchAsym(r, P=train_df["P_t_mps"], N=train_df["N_t_mps"], dist=dist)
    elif model_name == "B5.4":
        return NAGarchAsym(
            r, P=train_df["P_t_speech"], N=train_df["N_t_speech"], dist=dist
        )
    elif model_name == "B5.5":
        # Bulletins-only stance (Monthly + Economic Bulletins)
        return NAGarchAsym(
            r, P=train_df["P_t_bulletins"], N=train_df["N_t_bulletins"], dist=dist
        )
    elif model_name == "B5.6":
        # τ=0 confidence threshold (no filter on classifier confidence)
        return NAGarchAsym(
            r, P=train_df["P_t_t000"], N=train_df["N_t_t000"], dist=dist
        )
    elif model_name == "B5.7":
        # τ=0.80 confidence threshold (high-confidence sentences only)
        return NAGarchAsym(
            r, P=train_df["P_t_t080"], N=train_df["N_t_t080"], dist=dist
        )
    raise ValueError(f"Unknown model name: {model_name}")


def is_nagarch(model_name):
    return model_name.startswith("B2.") or model_name.startswith("B5.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SINGLE-MODEL OOS RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def build_origins(df_full, in_sample_end_idx, week_step):
    """
    List of global row indices for the forecast origins.

    First origin = last in-sample row (forecasts the first OOS day).
    Subsequent origins step by week_step trading days through the OOS period.
    """
    oos_idx = df_full.index[df_full["period"] == "outsample"].tolist()
    if not oos_idx:
        return []
    last_oos_idx = oos_idx[-1]

    origins = [in_sample_end_idx]
    step = in_sample_end_idx + week_step
    while step <= last_oos_idx:
        origins.append(step)
        step += week_step
    return origins


def filter_train_for_nagarch(model_name, train_df):
    """
    Drop training rows where the news regressors are NaN. Important for
    rolling windows that step into early OOS — some columns (like
    P_mps_surprise) are 0 pre-first-MPS, but P_t can have NaN if the
    upstream series wasn't backfilled into pre-sample.
    """
    cols_needed = {
        "B2.1": ["S_t"],
        "B2.2": ["P_t", "N_t"],
        "B2.3": ["P_mps_surprise", "N_mps_surprise"],
        "B5.3": ["P_t_mps", "N_t_mps"],
        "B5.4": ["P_t_speech", "N_t_speech"],
        "B5.5": ["P_t_bulletins", "N_t_bulletins"],
        "B5.6": ["P_t_t000", "N_t_t000"],
        "B5.7": ["P_t_t080", "N_t_t080"],
    }.get(model_name, [])
    if cols_needed:
        train_df = train_df.dropna(subset=["log_return"] + cols_needed)
    else:
        train_df = train_df.dropna(subset=["log_return"])
    return train_df


def run_oos(df_full, model_name, scheme, horizon, dist=DIST,
            week_step=WEEK_STEP, max_origins=None):
    """
    Run OOS variance forecasting for one (model, scheme, horizon) combo.

    Returns: DataFrame with forecast / realized columns, one row per
    forecast origin.
    """
    is_mask = df_full["period"] == "insample"
    is_indices = np.where(is_mask)[0]
    in_sample_end_idx = int(is_indices[-1])
    in_sample_size = int(is_mask.sum())

    origins = build_origins(df_full, in_sample_end_idx, week_step)
    if max_origins is not None:
        origins = origins[:max_origins]

    log.info(f"  {model_name} | {scheme} | h={horizon} | "
             f"{len(origins)} forecast origins")

    rows = []
    warm_theta = None
    n_failed = 0
    t0 = time.time()

    for k, origin_idx in enumerate(origins):
        # Build training data
        if scheme == "RW":
            train_start = max(0, origin_idx - in_sample_size + 1)
        else:  # IW
            train_start = int(is_indices[0])
        train_full = df_full.iloc[train_start : origin_idx + 1]
        train = filter_train_for_nagarch(model_name, train_full)

        if len(train) < 100:
            n_failed += 1
            continue

        # Fit
        try:
            m = make_model(model_name, train, dist=dist)
            if is_nagarch(model_name) and warm_theta is not None:
                # Warm-start dimensions must match
                if hasattr(m, "_theta_init") and len(warm_theta) == len(m._theta_init()):
                    m.fit(n_restarts=0, warm_start={"theta": warm_theta})
                else:
                    m.fit(n_restarts=1)
            else:
                if is_nagarch(model_name):
                    m.fit(n_restarts=1)
                else:
                    m.fit()
            if is_nagarch(model_name):
                warm_theta = m._theta_opt.copy()
        except Exception as e:
            log.warning(f"    fit failed at origin {origin_idx} "
                        f"({df_full.iloc[origin_idx]['date'].date()}): {e}")
            n_failed += 1
            continue

        # Forecast
        try:
            forecast_path = m.forecast_variance(h=horizon)
        except Exception as e:
            log.warning(f"    forecast failed at origin {origin_idx}: {e}")
            n_failed += 1
            continue

        if horizon == 1:
            forecast_var = float(forecast_path[0])
        else:
            # Cumulative variance over next h trading days
            forecast_var = float(np.sum(forecast_path))

        # Realized target
        target_idx = origin_idx + 1
        if target_idx >= len(df_full):
            continue
        if horizon == 1:
            realized = float(df_full.iloc[target_idx]["sq_return"])
            abs_return_target = float(df_full.iloc[target_idx]["abs_return"])
        else:
            realized = float(df_full.iloc[target_idx]["sq_return_5d"])
            # For h=5 abs_return target = sum of next 5 abs_returns
            abs_return_target = float(
                df_full.iloc[target_idx : target_idx + horizon]["abs_return"].sum()
            )

        rows.append({
            "origin_date":          df_full.iloc[origin_idx]["date"],
            "forecast_target_date": df_full.iloc[target_idx]["date"],
            "model":                model_name,
            "scheme":               scheme,
            "horizon":              horizon,
            "forecast_variance":    forecast_var,
            "realized_variance":    realized,
            "abs_return_target":    abs_return_target,
            "n_train":              len(train),
        })

        if (k + 1) % max(1, len(origins) // 10) == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(origins) - k - 1) / max(k + 1, 1)
            log.info(f"    [{k+1:>4}/{len(origins)}]  "
                     f"elapsed {elapsed/60:5.1f}m  eta {eta/60:5.1f}m")

    log.info(f"  {model_name} | {scheme} | h={horizon}: done in "
             f"{(time.time()-t0)/60:.1f}m  ({n_failed} failures)")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Run only these models (e.g. --models B1.1 B2.2)")
    parser.add_argument("--gaussian", action="store_true",
                        help="Gaussian innovations")
    parser.add_argument("--quick", action="store_true",
                        help="Use only the first 20 forecast origins (smoke test)")
    args = parser.parse_args()

    dist = "normal" if args.gaussian else "studentst"
    max_origins = 20 if args.quick else None

    df_full = pd.read_csv(MASTER_CSV, parse_dates=["date"])
    log.info("=" * 70)
    log.info("OUT-OF-SAMPLE FORECASTING")
    log.info(f"  master CSV: {len(df_full):,} rows  "
             f"(presample/insample/outsample = "
             f"{(df_full['period']=='presample').sum()}/"
             f"{(df_full['period']=='insample').sum()}/"
             f"{(df_full['period']=='outsample').sum()})")
    log.info(f"  innovations: {dist}")
    log.info(f"  week_step: {WEEK_STEP} (forecast origin every {WEEK_STEP} trading days)")
    if args.quick:
        log.info(f"  QUICK MODE: max_origins=20")
    log.info("=" * 70)

    # Build the full task list
    models_to_run = args.models if args.models else (MAIN_MODELS + ROBUSTNESS_MODELS)
    tasks = []
    for m in models_to_run:
        for s in schemes_for(m):
            for h in HORIZONS:
                tasks.append((m, s, h))

    log.info(f"\nTotal (model, scheme, horizon) combos to run: {len(tasks)}")
    for t in tasks:
        log.info(f"  {t}")

    t0 = time.time()
    for i, (model_name, scheme, horizon) in enumerate(tasks):
        log.info(f"\n──[{i+1}/{len(tasks)}] {model_name}  scheme={scheme}  h={horizon} ──")
        result = run_oos(df_full, model_name, scheme, horizon, dist=dist,
                         max_origins=max_origins)
        out_path = OOS_DIR / f"{model_name}_{scheme}_h{horizon}.csv"
        result.to_csv(out_path, index=False)
        log.info(f"  → {out_path}  ({len(result)} rows)")

    elapsed_min = (time.time() - t0) / 60
    log.info(f"\n{'='*70}")
    log.info(f"OOS FORECASTING COMPLETE — {elapsed_min:.1f} minutes total")
    log.info(f"  Outputs in: {OOS_DIR}")
    log.info(f"  Next: python evaluate.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
