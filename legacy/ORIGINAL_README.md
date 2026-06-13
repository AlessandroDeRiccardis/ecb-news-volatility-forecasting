# ECB Sentiment & Volatility Forecasting

Replication code for "Volatility Forecasting and ECB News Sentiment" — a non-replication study testing whether the Sadik, Date & Mitra (2018) NA-GARCH framework, applied to ECB communication scored with a stance-classification LLM (FOMC-RoBERTa), improves volatility forecasting for the Euro Stoxx 50.

**Headline result:** the news term does not improve forecasts beyond standard GARCH benchmarks. The result is robust to multiple stance-encoding choices, three different surprise-extraction strategies, source decomposition (MPS-only / speeches-only / bulletins-only), confidence thresholds, vol-proxy choice, and innovations distribution.

## Reference papers

- Sadik, Z., Date, P., & Mitra, G. (2018). News Augmented GARCH(1,1) Model for Volatility Prediction. *IMA Journal of Management Mathematics*. (Primary methodology — Eq. 3.10/3.11.)
- Smales, L. A. (2014). Time-varying relationship of news sentiment, implied volatility and stock returns. SSRN id 2186267. (Asymmetric news effects on VIX.)
- Bodilsen, S. T., & Lunde, A. (2024). Exploiting News Analytics for Volatility Forecasting. *Journal of Applied Econometrics*. (Macro vs. firm-specific news.)
- Bernoth, K. (2026). Dovish Coos or Hawkish Screech? Identifying ECB Communication Shocks. DIW Berlin Discussion Paper. (ECB-fine-tuned stance model and residualization-based surprise extraction.)
- Shah, A., Paturi, S., & Chava, S. (2023). Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis. ACL 2023. (FOMC-RoBERTa stance classifier — `gtfintechlab/FOMC-RoBERTa`.)

## Repository structure

Scripts at the top level are numbered by pipeline stage (01 → 05). Three files keep un-numbered names because they are imported by the orchestrator: `models.py`, `forecast_oos.py`, and `run_main_pipeline.py`.

```
ecb_sentiment/
│
├── README.md                              ← this file (the replication guide)
│
├── 01a_download_market_data.py            STAGE 1 — Eurostoxx 50, VSTOXX, ECB AAA 10y yield
├── 01b_download_controls.py                          EUR/USD, VIX, Brent (yfinance)
│
├── 02_ecb_collection.py                   STAGE 2 — ECB document collection (foedb API)
│
├── 03a_score_sentences.py                 STAGE 3 — Sentence-level FOMC-RoBERTa stance (~7h CPU)
├── 03b_merge_sources.py                              Merge legacy + freshly scored sentence CSVs
│
├── 04a_build_daily_series.py              STAGE 4 — Document- and day-level aggregation, Sadik rescaling
├── 04b_diagnostics.py                                Optional pre-modeling sanity-check plots
│
├── 05a_prepare_master_dataset.py          STAGE 5 — Build model_data_master.csv
├── models.py                                         Model classes (GARCH / GJR / EGARCH / NA-GARCH-net / NA-GARCH-asym)
├── forecast_oos.py                                   OOS forecasting machinery (imported by orchestrator)
├── 05b_estimate_in_sample.py                         Standalone in-sample MLE script (legacy entry)
├── 05c_evaluate.py                                   Standalone OOS evaluation (legacy entry)
├── 05d_extract_surprises_bernoth.py                  Bernoth-style residualization (appendix robustness)
├── run_main_pipeline.py                   ★         FINAL ORCHESTRATOR — what to run for full replication ★
│
├── 06_make_report_artifacts.py            STAGE 6 — LaTeX tables + figures for the paper
│
├── market_data/                           Raw market-data CSVs
├── output/                                All intermediate + final outputs
└── old/                                   Archived deprecated scripts and outputs
```

## Replication steps

### 1. Environment

```bash
cd ecb_sentiment
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scipy arch transformers torch yfinance matplotlib tqdm
```

### 2. Market & control data — Stage 1

```bash
python 01a_download_market_data.py     # Eurostoxx 50 prices/returns, VSTOXX, ECB AAA 10y yield
python 01b_download_controls.py        # EUR/USD, VIX, Brent (yfinance)
```

Then **manually** download the ECB AAA 1-year yield from the [ECB Data Portal](https://data.ecb.europa.eu) ("Yields for 1Y maturity") and save to `market_data/ecb_aaa_1y_yield.csv` with columns `date,yield_1y_aaa`.

### 3. ECB document collection — Stage 2

```bash
python 02_ecb_collection.py
```

Downloads ~1,160 ECB documents (Monetary Policy Statements, Monthly + Economic Bulletins, speeches by President and Chief Economist) into `output/ecb_raw_texts_clean/`. Q&A and PDF boilerplate stripped at this stage. About 30 minutes.

### 4. NLP scoring — Stage 3 (~7 hours CPU)

```bash
python 03a_score_sentences.py
python 03b_merge_sources.py            # only if mixing freshly scored bulletins with legacy speeches/MPS
```

For test runs use `python 03a_score_sentences.py --limit 20`. To resume after interruption: `--resume`.

### 5. Daily aggregation — Stage 4

```bash
python 04a_build_daily_series.py       # produces all_sources / mps_only / speeches_only / bulletins_only daily CSVs
python 04b_diagnostics.py              # optional — pre-modeling sanity-check plots
```

### 6. Master dataset — Stage 5

```bash
python 05a_prepare_master_dataset.py
```

Produces `output/model_data_master.csv` — the single CSV that all modeling scripts consume. Contains daily Euro Stoxx 50 returns + rolling vol + stance series at all sources/thresholds + financial-state controls + period flags.

### 7. Modeling — final pipeline

```bash
python run_main_pipeline.py            # full run, ~2 hours (in-sample + OOS + evaluation)

# Faster modes for verification:
python run_main_pipeline.py --in-sample-only      # ~30 sec — sanity check
python run_main_pipeline.py --skip-oos            # in-sample + diagnostics, reuse OOS forecasts
python run_main_pipeline.py --quick-oos           # 20 OOS origins (~10 min)
python run_main_pipeline.py --include-appendix    # adds B2.3 surprise spec
```

The orchestrator writes `output/final_results_summary.txt` (the consolidated report) plus underlying CSVs (`final_in_sample_*.csv`, `final_oos_*.csv`, `final_residual_diagnostics.csv`, `final_forecast_combination.csv`).

### 8. Bernoth-residualized surprises (appendix)

```bash
python 05d_extract_surprises_bernoth.py
```

Reports in-sample comparison of NA-GARCH-asym with Bernoth-style residualized inputs vs. raw levels. Confirms the null result holds under residualization.

### 9. Report artifacts — Stage 6

```bash
python 06_make_report_artifacts.py             # tables + figures
python 06_make_report_artifacts.py --no-plots  # tables only
```

Reads the `final_*.csv` files produced by `run_main_pipeline.py` and writes paper-ready artifacts to `output/report/`:

- `tables.tex` — all paper tables in LaTeX (booktabs format)
- `figure_1_returns_vol.png` — Euro Stoxx 50 returns + 20-day realized vol
- `figure_2_stance_series.png` — Daily P_t / N_t / S_t with crisis annotations
- `figure_3_insample_fit.png` — In-sample fitted volatility (GARCH / GJR / NA-GARCH-asym) vs realized
- `figure_4_oos_forecasts.png` — OOS one-step volatility forecasts (rolling window)
- `figure_5_news_scaling.png` — Implied f(P, N) news-scaling factor under fitted B2.2

## Final model suite

### Main spec (5 models, OOS evaluated × 4 schemes = 20 forecast runs)

| ID    | Name                              | Description                                                 |
|-------|-----------------------------------|-------------------------------------------------------------|
| B1.1  | GARCH(1,1)                        | Simplest baseline, no news                                  |
| B1.2  | GJR-GARCH(1,1)                    | Leverage benchmark, no news                                 |
| B1.3  | EGARCH(1,1)                       | Sadik et al. (2018) primary benchmark                       |
| B2.1  | NA-GARCH-net                      | Sadik replication, single net stance S_t = P_t + N_t        |
| B2.2  | NA-GARCH-asym                     | Sadik primary spec, P_t and N_t separate (κ ≠ γ)            |

### Robustness suite (in-sample + selected OOS)

| ID    | Name                              | Notes                                                       |
|-------|-----------------------------------|-------------------------------------------------------------|
| B5.1  | abs_return target                 | Re-evaluates B2.2 forecasts on volatility scale             |
| B5.3  | NA-GARCH-asym MPS-only            | Only MPS stance enters P_t, N_t                             |
| B5.4  | NA-GARCH-asym speeches-only       | Only speeches enter                                         |
| B5.5  | NA-GARCH-asym bulletins-only      | Only Monthly + Economic Bulletins enter                     |
| B5.6  | NA-GARCH-asym at τ=0              | No confidence-threshold filter on classifier output         |
| B5.7  | NA-GARCH-asym at τ=0.80           | High-confidence sentences only                              |
| B5.8  | NA-GARCH-asym Gaussian            | Gaussian innovations instead of Student-t                   |

### Appendix (in-sample only — reported separately, all underperform B2.2)

| ID    | Name                                            |
|-------|-------------------------------------------------|
| B2.3  | NA-GARCH-asym, MPS-vs-intra-period-baseline surprises (decay half-life 5d) |
| B6.1  | NA-GARCH-asym, Bernoth residuals with AR(2) lags |
| B6.2  | NA-GARCH-asym, Bernoth residuals without AR lags |

### Not run (tested earlier and rejected)

- **NA-GARCH-Ht** (regime indicator × stance interaction). H_t × stance interaction adds at most +1.06 LL units over B2.2, with Hansen (1999) fixed-regressor bootstrap returning p = 0.46. Documented as one sentence in the paper; not in the modeling code.

## Sample windows

| Period      | Range                       | n trading days | Use                                       |
|-------------|-----------------------------|----------------|-------------------------------------------|
| pre-sample  | 2007-04 → 2007-12           | 187            | GARCH initialization only                 |
| in-sample   | 2008-01-01 → 2018-12-31     | 2,754          | Parameter estimation, AIC, Sadik rescaling |
| out-sample  | 2019-01-01 → 2023-12-31     | 1,260          | Rolling- and increasing-window OOS         |

OOS sub-samples for time-varying-effect analysis: `2019_normal`, `2020_2021_covid`, `2022_2023_infl`.

## Key methodological choices

- **Stance scoring** uses `gtfintechlab/FOMC-RoBERTa` (Shah et al. 2023) at sentence level. Classifier returns 3-class probability vector (dovish / hawkish / neutral); main spec uses confidence threshold τ = 0.50.
- **Document aggregation:** P_doc_dovish = Σ dovish-prob over qualifying sentences / total sentences. Total-sentence denominator (rather than non-neutral) decouples the two directional series — empirical correlation with this denominator is +0.34 rather than the −1 it would be under non-neutral normalization.
- **Daily aggregation:** equal-weight mean across documents on multi-doc days, persist-to-next-event smoothing, Sadik et al. (2018) Eq. 3.2 in-sample-max rescaling so P_t ∈ [0, 1] and N_t ∈ [−1, 0] in-sample.
- **NA-GARCH variance equation** (Sadik et al. Eq. 3.11):
  σ²_t = f(P_{t−1}, N_{t−1}) · (ω + α ε²_{t−1} + β σ²_{t−1})
  with f(P, N) = a + 0.5 b · [tanh(κP/2) − tanh(γN/2)]
  and constraints α + β < 1, 0.5 ≤ a + b ≤ 2.

## What to send to teammates

For a self-contained replication package, include:

**Code (required):**
- `README.md`
- All `*.py` files at the top level (15 scripts)

**Inputs (required):**
- `market_data/` — full folder with all CSVs

**Final outputs (required for verification):**
- `output/final_results_summary.txt` — consolidated report
- `output/final_in_sample_main.csv`
- `output/final_in_sample_robust.csv`
- `output/final_residual_diagnostics.csv`
- `output/final_oos_main.csv`
- `output/final_oos_dm.csv`
- `output/final_oos_mcs.csv`
- `output/final_oos_subsample.csv`
- `output/final_oos_absreturn.csv`
- `output/final_forecast_combination.csv`
- `output/model_data_master.csv` — the modeling dataset

**Intermediate outputs (recommended for full audit):**
- `output/sentiment_sentence_level_v4.csv` — every sentence with its 3-class probability vector (for auditing the stance scoring)
- `output/sentiment_document_level_v4.csv` — document-level dovish/hawkish/intensity scores
- `output/sentiment_daily_all_sources_v4.csv` — daily series feeding the master
- `output/sentiment_daily_mps_only_v4.csv`, `..._speeches_only_v4.csv`, `..._bulletins_only_v4.csv` — source-decomposition daily series
- `output/ecb_documents_master.csv` — index of all ECB documents collected
- `output/oos_forecasts/` — folder with one CSV per (model, scheme, horizon)

**Optional (large; share via cloud rather than email):**
- `output/ecb_raw_texts_clean/` — every cleaned ECB document (~1,160 files)

**Skip:**
- `venv/` — environment is recreated by teammates from the `pip install` line in this README
- `old/` — archived material, not part of the current pipeline
- `output/*.log` — logs are regenerated on rerun

## Outputs reference

`output/final_results_summary.txt` is the consolidated human-readable report. It contains:

- Table 1A — In-sample MLE (main suite)
- Table 1B — In-sample MLE (robustness suite)
- Table 1C — Ljung-Box residual diagnostics (lags 5, 10, 20; on residuals and squared residuals)
- Table 2  — OOS QLIKE / RMSE (main + robustness, RW/IW × h=1/5)
- Table 3  — Diebold-Mariano tests vs B1.1 GARCH and B1.2 GJR
- Table 4  — Model Confidence Set at 90% (using `arch.bootstrap.MCS`, T_max statistic)
- Table 5  — Sub-sample OOS analysis (2019 / 2020–21 / 2022–23)
- Table 6  — B5.1 abs_return-target robustness (vol-scale comparison)
- Table 7  — Forecast combination (B1.2 + B2.2)
- Table 8  — Economic significance of news scaling under B2.2
- Appendix — surprise specifications (only with `--include-appendix`)

Each table is also written as a standalone CSV (`final_*.csv`) for direct LaTeX-formatting.
