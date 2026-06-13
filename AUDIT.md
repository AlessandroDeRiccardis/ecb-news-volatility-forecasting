# Project Audit

Audit completed on 2026-06-13 before restructuring.

See [`legacy/FILE_INVENTORY.md`](legacy/FILE_INVENTORY.md) for the detailed
file-by-file data flow and disposition.

## Scientific Workflow

1. Download Euro Stoxx 50 prices, returns, volatility proxies, and controls.
2. Collect and clean ECB Monetary Policy Statements, bulletins, and selected speeches.
3. Segment cleaned documents and classify sentences with FOMC-RoBERTa.
4. Aggregate to document scores, equal-weight event days, persist to the next
   event, and rescale by in-sample maxima.
5. Align stance with market returns and construct the master dataset.
6. Estimate GARCH, GJR-GARCH, EGARCH, NA-GARCH-net, and NA-GARCH-asym models.
7. Produce weekly-origin rolling/increasing-window forecasts at horizons 1 and 5.
8. Evaluate with QLIKE, RMSE, DM tests, MCS, forecast combinations, sub-samples,
   and robustness variants.

## Supplied Files And Disposition

| Original artifact | Purpose | Disposition |
|---|---|---|
| `01a_download_market_data.py`, `01b_download_controls.py` | Market acquisition | archived; logic documented |
| `02_ecb_collection.py` | ECB API/PDF/HTML collection and cleaning | archived; raw corpus missing |
| `03a_score_sentences.py` | FOMC-RoBERTa sentence scoring | archived; model/corpus required |
| `03b_merge_sources.py` | merge legacy and fresh scores | archived; known broken import |
| `04a_build_daily_series.py`, `04b_diagnostics.py` | stance aggregation and diagnostics | core aggregation refactored |
| `05a_prepare_master_dataset.py` | master dataset construction | archived; processed snapshot tracked |
| `models.py`, `forecast_oos.py`, `run_main_pipeline.py` | authoritative modeling workflow | core logic promoted to package |
| `05b_estimate_in_sample.py`, `05c_evaluate.py` | older standalone model/evaluation entries | archived |
| `05d_extract_surprises_bernoth.py` | Bernoth-style robustness | helper refactored; original archived |
| `06_make_report_artifacts.py` | paper figures and tables | final outputs tracked |
| `model_data_master.csv` | processed modeling data | moved to `data/processed/` |
| `Macroeconometrics.pdf` | final paper | moved to `paper/` |
| `Outputs/` | supplied final results | moved to `reports/` |

## Findings

- No notebooks or configuration files were supplied.
- No absolute local paths were found in the Python code.
- The public paper copy was sanitized to remove student IDs and personal
  emails while preserving authorship and scientific content.
- Raw market files, ECB documents, sentence scores, daily stance files,
  per-origin forecasts, and logs were not supplied.
- Model labels vary between code (`B2.1`, `B2.2`, `B5.x`) and paper
  (`NA1.1`, `NA1.2`, `R1.x`). Public documentation maps both conventions.
- The supplied results match the paper's main numerical claims.
- A deterministic refactored re-fit found a marginally higher local optimum for
  `B2.2` (`loglik=8195.289`) than the supplied main table (`8194.981`). This
  changes neither the AIC ranking nor the headline null and exposes the
  expected local-optimum sensitivity of a nonlinear NA-GARCH likelihood.

## Empirical Result

The result is a robust null: ECB stance does not improve out-of-sample Euro
Stoxx 50 volatility forecasts beyond asymmetric GARCH benchmarks. EGARCH has
the lowest QLIKE in every main OOS cell; GJR and EGARCH are the only models
retained by the 90% Model Confidence Set; the optimal forecast-combination
weight on NA-GARCH-asym is zero in all four cells.
