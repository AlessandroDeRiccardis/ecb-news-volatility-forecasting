"""
06_make_report_artifacts.py — generate LaTeX tables and plots for the paper
==========================================================================

Reads the orchestrator's final_*.csv outputs and the master dataset, produces
a single LaTeX file with all paper tables plus the standard set of plots:

    output/report/tables.tex            All tables in LaTeX (booktabs format)
    output/report/figure_1_returns_vol.png       Returns + realized vol over time
    output/report/figure_2_stance_series.png     P_t / N_t / S_t with crisis markers
    output/report/figure_3_insample_fit.png      Fitted σ² vs realized r² (in-sample)
    output/report/figure_4_oos_forecasts.png     OOS σ² vs realized r² (RW, h=1)
    output/report/figure_5_news_scaling.png      Implied f(P, N) under fitted B2.2

USAGE
-----
    python 06_make_report_artifacts.py
    python 06_make_report_artifacts.py --no-plots   # tables only

Run AFTER `python run_main_pipeline.py` so the final_*.csv files exist.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from models import GARCH11, GJRGARCH, EGARCH, NAGarchAsym, NAGarchNet


OUTPUT_DIR = Path("./output")
REPORT_DIR = OUTPUT_DIR / "report"
OOS_DIR    = OUTPUT_DIR / "oos_forecasts"

REPORT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "make_report_artifacts.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# Crisis/event annotations to mark on the time-series plots
CRISIS_EVENTS = [
    ("Lehman",            "2008-09-15"),
    ("Sov. debt crisis",  "2010-05-10"),
    ("'whatever it takes'","2012-07-26"),
    ("APP launch",        "2015-01-22"),
    ("COVID emergency",   "2020-03-18"),
    ("First post-COVID hike", "2022-07-21"),
]

INSAMPLE_END = pd.Timestamp("2018-12-31")
OOS_START    = pd.Timestamp("2019-01-01")


# ─────────────────────────────────────────────────────────────────────────────
# 1. TABLES (LaTeX)
# ─────────────────────────────────────────────────────────────────────────────

def df_to_latex(df, caption, label, float_format="%.4f", index=False):
    """Wrap pandas to_latex with booktabs styling and a sensible default."""
    return df.to_latex(
        index=index,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=True,
        column_format=None,
    )


def build_descriptive_stats():
    """Descriptive statistics for returns, realized vol, stance series."""
    df = pd.read_csv(OUTPUT_DIR / "model_data_master.csv", parse_dates=["date"])
    ins = df[df["period"] == "insample"]
    oos = df[df["period"] == "outsample"]

    rows = []
    for label, sub in [("In-sample (2008–2018)", ins), ("OOS (2019–2023)", oos), ("Full", df[df["period"] != "presample"])]:
        rows.append({
            "Sample":  label,
            "n":       len(sub),
            "Return mean (%)":      sub["log_return"].mean() * 100,
            "Return SD (%)":        sub["log_return"].std()  * 100,
            "Return skew":          float(sub["log_return"].skew()),
            "Return kurt":          float(sub["log_return"].kurt()),
            "Realized vol (annu.)": sub["log_return"].std() * np.sqrt(252) * 100,
            "P_t mean":             float(sub["P_t"].mean(skipna=True)),
            "N_t mean":             float(sub["N_t"].mean(skipna=True)),
            "S_t mean":             float(sub["S_t"].mean(skipna=True)),
        })
    return pd.DataFrame(rows)


def write_all_tables():
    log.info("  Building LaTeX tables ...")
    sections = []

    # ── Table 0: Descriptive statistics ────────────────────────────────────
    desc = build_descriptive_stats()
    sections.append(df_to_latex(
        desc.round(4),
        caption="Descriptive statistics of returns and stance series.",
        label="tab:desc_stats",
    ))

    # ── Table 1A: In-sample main suite ─────────────────────────────────────
    if (OUTPUT_DIR / "final_in_sample_main.csv").exists():
        t1a = pd.read_csv(OUTPUT_DIR / "final_in_sample_main.csv")
        sections.append(df_to_latex(
            t1a[["model", "n_obs", "loglik", "aic", "n_params"]].round(3),
            caption="In-sample MLE estimates, main suite. Lower AIC is better.",
            label="tab:insample_main",
        ))

    # ── Table 1B: In-sample robustness suite ───────────────────────────────
    if (OUTPUT_DIR / "final_in_sample_robust.csv").exists():
        t1b = pd.read_csv(OUTPUT_DIR / "final_in_sample_robust.csv")
        sections.append(df_to_latex(
            t1b[["model", "dist", "n_obs", "loglik", "aic"]].round(3),
            caption="In-sample MLE estimates, robustness suite.",
            label="tab:insample_robust",
        ))

    # ── Table 1C: Ljung-Box residual diagnostics ───────────────────────────
    if (OUTPUT_DIR / "final_residual_diagnostics.csv").exists():
        t1c = pd.read_csv(OUTPUT_DIR / "final_residual_diagnostics.csv")
        sections.append(df_to_latex(
            t1c.round(3),
            caption="Ljung-Box test on standardized residuals "
                    "and squared standardized residuals (main suite). "
                    "$H_0$: no autocorrelation.",
            label="tab:ljung_box",
        ))

    # ── Table 2: OOS QLIKE / RMSE ──────────────────────────────────────────
    if (OUTPUT_DIR / "final_oos_main.csv").exists():
        t2 = pd.read_csv(OUTPUT_DIR / "final_oos_main.csv")
        sections.append(df_to_latex(
            t2.sort_values(["scheme", "horizon", "QLIKE"]).round(6),
            caption="Out-of-sample evaluation: QLIKE and RMSE by model, "
                    "scheme (rolling window / increasing window) and horizon. "
                    "Lower QLIKE is better.",
            label="tab:oos_main",
        ))

    # ── Table 3: DM tests ──────────────────────────────────────────────────
    if (OUTPUT_DIR / "final_oos_dm.csv").exists():
        t3 = pd.read_csv(OUTPUT_DIR / "final_oos_dm.csv")
        sections.append(df_to_latex(
            t3.round(4),
            caption="Diebold-Mariano tests of equal predictive accuracy. "
                    "$\\Delta_{QL} =$ QLIKE(model) $-$ QLIKE(baseline); "
                    "negative values favour the candidate model. "
                    "Newey-West HAC standard errors with bandwidth $h-1$ for $h>1$.",
            label="tab:dm",
        ))

    # ── Table 4: MCS at 90% ────────────────────────────────────────────────
    if (OUTPUT_DIR / "final_oos_mcs.csv").exists():
        t4 = pd.read_csv(OUTPUT_DIR / "final_oos_mcs.csv")
        sections.append(df_to_latex(
            t4,
            caption="Hansen-Lunde-Nason (2011) Model Confidence Set at 90\\% confidence. "
                    "Models in the MCS cannot be statistically eliminated as inferior.",
            label="tab:mcs",
        ))

    # ── Table 5: Sub-sample OOS ────────────────────────────────────────────
    if (OUTPUT_DIR / "final_oos_subsample.csv").exists():
        t5 = pd.read_csv(OUTPUT_DIR / "final_oos_subsample.csv")
        sections.append(df_to_latex(
            t5.sort_values(["subsample", "scheme", "horizon", "QLIKE"]).round(6),
            caption="Sub-sample OOS evaluation across regimes: "
                    "2019 (normal), 2020--21 (COVID), 2022--23 (inflation shock).",
            label="tab:subsample",
        ))

    # ── Table 6: abs_return target ─────────────────────────────────────────
    if (OUTPUT_DIR / "final_oos_absreturn.csv").exists():
        t6 = pd.read_csv(OUTPUT_DIR / "final_oos_absreturn.csv")
        sections.append(df_to_latex(
            t6.round(4),
            caption="Robustness to volatility-proxy choice: B5.1 reuses B2.2 forecasts "
                    "evaluated against $|r_t|$ (vol scale). Same QLIKE / RMSE definitions; "
                    "forecasts taken as $\\sqrt{\\hat{\\sigma}^2_t}$ for unit consistency.",
            label="tab:absreturn",
        ))

    # ── Table 7: Forecast combination ──────────────────────────────────────
    if (OUTPUT_DIR / "final_forecast_combination.csv").exists():
        t7 = pd.read_csv(OUTPUT_DIR / "final_forecast_combination.csv")
        sections.append(df_to_latex(
            t7.round(6),
            caption="Forecast combination test: (1$-w$) $\\cdot$ B1.2 + $w \\cdot$ B2.2. "
                    "$w^*$ near zero indicates B2.2 carries no orthogonal information vs GJR.",
            label="tab:combination",
        ))

    # ── Header + assemble ──────────────────────────────────────────────────
    header = (
        "% Auto-generated by 06_make_report_artifacts.py\n"
        "% Each table is in booktabs format. \\usepackage{booktabs} required.\n"
        "% \\usepackage{caption,amsmath} also recommended.\n\n"
    )
    text = header + "\n\n".join(sections) + "\n"
    out = REPORT_DIR / "tables.tex"
    out.write_text(text)
    log.info(f"    → {out} ({len(sections)} tables)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PLOT 1 — Returns and realized volatility over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_returns_vol():
    log.info("  Plot 1: returns + realized vol ...")
    df = pd.read_csv(OUTPUT_DIR / "model_data_master.csv", parse_dates=["date"])
    # Compute rolling vol on the fly (no longer carried in master to avoid
    # redundancy — see clean-up note in prepare_master_dataset).
    df = df.sort_values("date").reset_index(drop=True)
    df["roll_vol_20d_calc"] = df["log_return"].rolling(20, min_periods=20).std()
    df = df[df["period"] != "presample"].copy()
    df["realized_vol_20d"] = df["roll_vol_20d_calc"] * np.sqrt(252) * 100  # annualized %

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    axes[0].plot(df["date"], df["log_return"] * 100, linewidth=0.6, color="#264653")
    axes[0].axvline(INSAMPLE_END, color="grey", linestyle="--", linewidth=0.7)
    axes[0].set_ylabel("Daily log return (%)")
    axes[0].set_title("Euro Stoxx 50: daily returns and 20-day realized volatility")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["date"], df["realized_vol_20d"], linewidth=0.9, color="#e76f51")
    axes[1].axvline(INSAMPLE_END, color="grey", linestyle="--", linewidth=0.7,
                    label="In-sample / OOS split")
    axes[1].set_ylabel("Annualized vol. (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = REPORT_DIR / "figure_1_returns_vol.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLOT 2 — Stance series over time, with crisis markers
# ─────────────────────────────────────────────────────────────────────────────

def plot_stance_series():
    log.info("  Plot 2: stance series ...")
    df = pd.read_csv(OUTPUT_DIR / "model_data_master.csv", parse_dates=["date"])
    df = df[df["period"] != "presample"].dropna(subset=["P_t", "N_t", "S_t"]).copy()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], df["P_t"], color="#2a9d8f", linewidth=0.9, label="$P_t$ (dovish)")
    ax.plot(df["date"], df["N_t"], color="#e76f51", linewidth=0.9, label="$N_t$ (hawkish, negated)")
    ax.plot(df["date"], df["S_t"], color="#264653", linewidth=0.5, alpha=0.5, label="$S_t = P_t + N_t$")
    ax.axhline(0.0, color="black", linewidth=0.4)
    ax.axvline(INSAMPLE_END, color="grey", linestyle="--", linewidth=0.7)

    # Annotate crisis events
    y0 = ax.get_ylim()[0]
    for label, date_s in CRISIS_EVENTS:
        d = pd.Timestamp(date_s)
        if df["date"].min() <= d <= df["date"].max():
            ax.axvline(d, color="grey", linewidth=0.4, alpha=0.7, linestyle=":")
            ax.text(d, y0 + 0.02, label, rotation=90, fontsize=7, va="bottom",
                    ha="right", alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("Sadik-rescaled stance")
    ax.set_title("Daily ECB stance series (FOMC-RoBERTa, $\\tau = 0.5$, persist-to-next-event)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = REPORT_DIR / "figure_2_stance_series.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT 3 — In-sample fitted vs realized
# ─────────────────────────────────────────────────────────────────────────────

def plot_insample_fit():
    log.info("  Plot 3: in-sample fitted vs realized ...")
    df = pd.read_csv(OUTPUT_DIR / "model_data_master.csv", parse_dates=["date"])
    ins = df[df["period"] == "insample"].dropna(subset=[
        "log_return", "P_t", "N_t", "S_t"
    ]).reset_index(drop=True)

    # Refit a small subset of models for plotting (cheap; ~20 sec total)
    fits = {}
    log.info("    refitting B1.1, B1.2, B2.2 for plotting ...")
    fits["B1.1 GARCH"]   = GARCH11(ins["log_return"], dist="studentst").fit()
    fits["B1.2 GJR"]     = GJRGARCH(ins["log_return"], dist="studentst").fit()
    fits["B2.2 NA-GARCH-asym"] = NAGarchAsym(
        ins["log_return"], P=ins["P_t"], N=ins["N_t"], dist="studentst"
    ).fit(n_restarts=2)

    # Convert to annualized vol (%)
    realized = ins["log_return"].rolling(20, min_periods=20).std() * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ins["date"], realized, color="grey", linewidth=0.6, alpha=0.6,
            label="20-day realized vol")
    colors = {"B1.1 GARCH": "#264653", "B1.2 GJR": "#e76f51", "B2.2 NA-GARCH-asym": "#2a9d8f"}
    for name, m in fits.items():
        sigma2 = m.conditional_variance()
        ann_vol = np.sqrt(sigma2) * np.sqrt(252) * 100
        ax.plot(ins["date"], ann_vol, color=colors[name], linewidth=0.7, label=name)

    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("In-sample fitted volatility vs 20-day realized (annualized)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = REPORT_DIR / "figure_3_insample_fit.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT 4 — OOS forecasts vs realized (RW, h=1)
# ─────────────────────────────────────────────────────────────────────────────

def plot_oos_forecasts():
    log.info("  Plot 4: OOS forecasts (RW, h=1) ...")
    plot_models = [("B1.1", "#264653"), ("B1.2", "#e76f51"),
                   ("B1.3", "#f4a261"), ("B2.2", "#2a9d8f")]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    realized_plotted = False
    for model, color in plot_models:
        p = OOS_DIR / f"{model}_RW_h1.csv"
        if not p.exists():
            log.warning(f"    {p} missing; skipping {model}")
            continue
        d = pd.read_csv(p, parse_dates=["forecast_target_date"]).sort_values("forecast_target_date")
        ann_real = np.sqrt(d["realized_variance"]) * np.sqrt(252) * 100
        ann_fore = np.sqrt(d["forecast_variance"]) * np.sqrt(252) * 100
        if not realized_plotted:
            ax.plot(d["forecast_target_date"], ann_real, color="grey",
                    linewidth=0.5, alpha=0.6, label="Realized $\\sqrt{r_t^2}$ (annualized)")
            realized_plotted = True
        ax.plot(d["forecast_target_date"], ann_fore, color=color, linewidth=0.8,
                label=model)

    ax.set_xlabel("Forecast target date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("Out-of-sample one-step volatility forecasts (rolling window)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = REPORT_DIR / "figure_4_oos_forecasts.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT 5 — News scaling factor f(P, N) under fitted B2.2
# ─────────────────────────────────────────────────────────────────────────────

def plot_news_scaling():
    log.info("  Plot 5: news scaling factor f(P, N) ...")
    df = pd.read_csv(OUTPUT_DIR / "model_data_master.csv", parse_dates=["date"])
    ins = df[df["period"] == "insample"].dropna(subset=["log_return", "P_t", "N_t"])
    m = NAGarchAsym(ins["log_return"], P=ins["P_t"], N=ins["N_t"],
                    dist="studentst").fit(n_restarts=2)
    p = m.params
    a, b, kappa, gamma = p["a"], p["b"], p["kappa"], p["gamma"]

    P_grid = np.linspace(0.0, 1.0, 60)
    N_grid = np.linspace(-1.0, 0.0, 60)
    PP, NN = np.meshgrid(P_grid, N_grid)
    F = a + 0.5 * b * (np.tanh(0.5 * kappa * PP) - np.tanh(0.5 * gamma * NN))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cf = ax.contourf(PP, NN, F, levels=20, cmap="RdBu_r")
    cs = ax.contour(PP, NN, F, levels=10, colors="black", linewidths=0.4, alpha=0.4)
    ax.clabel(cs, inline=1, fontsize=7, fmt="%.3f")
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("News scaling factor $f(P, N)$")
    ax.set_xlabel("$P_t$ (dovish, persisted+rescaled)")
    ax.set_ylabel("$N_t$ (hawkish, persisted+rescaled)")
    ax.set_title(
        f"Implied news scaling factor under fitted B2.2\n"
        f"$a={a:.3f}$, $b={b:.3f}$, $\\kappa={kappa:.3f}$, $\\gamma={gamma:.3f}$"
    )
    plt.tight_layout()
    out = REPORT_DIR / "figure_5_news_scaling.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation; tables only.")
    parser.add_argument("--no-tables", action="store_true",
                        help="Skip LaTeX table generation; figures only.")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("MAKE REPORT ARTIFACTS")
    log.info("=" * 70)

    if not args.no_tables:
        log.info("\n[1] LaTeX tables")
        write_all_tables()

    if not args.no_plots:
        log.info("\n[2] Figures")
        plot_returns_vol()
        plot_stance_series()
        plot_insample_fit()
        plot_oos_forecasts()
        plot_news_scaling()

    log.info("\n" + "=" * 70)
    log.info(f"ARTIFACTS WRITTEN → {REPORT_DIR}/")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
