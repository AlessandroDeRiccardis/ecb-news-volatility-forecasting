# Original File Inventory

This inventory records the supplied project's purpose, data flow, dependencies,
and disposition before restructuring.

| File | Purpose | Primary inputs | Primary outputs | Dependencies | Disposition |
|---|---|---|---|---|---|
| `01a_download_market_data.py` | Download Euro Stoxx 50, VSTOXX, ECB AAA 10y yield; construct returns and rolling volatility | Yahoo Finance, Stooq, STOXX, ECB APIs | legacy `market_data/*.csv` and summary | pandas, numpy, requests, yfinance | archived; acquisition logic retained |
| `01b_download_controls.py` | Download EUR/USD, VIX, Brent | Yahoo Finance | legacy control CSVs | yfinance | archived |
| `02_ecb_collection.py` | Collect, filter, deduplicate, and clean ECB communication | ECB FOEDB, ECB HTML/PDFs | document master and cleaned text corpus | requests, BeautifulSoup, pypdf, tqdm | archived; raw corpus absent |
| `03a_score_sentences.py` | Segment and score sentences with FOMC-RoBERTa | document master and cleaned text | sentence-level probability CSV | transformers, torch, pandas | archived; callable wrapper retained |
| `03b_merge_sources.py` | Merge earlier FOMC-scored rows with fresh monthly-bulletin rows | sentence CSVs and document master | canonical sentence-level CSV | pandas | archived; broken import documented |
| `04a_build_daily_series.py` | Aggregate sentence scores to document and daily stance series | canonical sentence-level CSV | document and source-specific daily stance CSVs | pandas, numpy | aggregation logic promoted |
| `04b_diagnostics.py` | Plot and summarize stance diagnostics | sentence, document, and daily stance CSVs | diagnostic figures and summary | matplotlib, pandas, numpy | archived |
| `05a_prepare_master_dataset.py` | Align returns, stance, surprises, and controls | market and daily stance CSVs | master modeling dataset | pandas, numpy | archived; final snapshot tracked |
| `models.py` | Implement benchmark and NA-GARCH model classes | return and stance arrays | fitted models and variance forecasts | arch, scipy, numba, numpy | promoted to package, original archived |
| `forecast_oos.py` | Re-estimate models at weekly OOS origins | master dataset | per-origin forecast CSVs | models, pandas, numpy | refactored; original archived |
| `05b_estimate_in_sample.py` | Older standalone model-estimation entry | master dataset | in-sample estimates and summary | models, pandas, numpy | archived |
| `05c_evaluate.py` | Older standalone forecast evaluation | OOS forecast CSVs and master dataset | evaluation tables and figures | arch, scipy, matplotlib | archived |
| `05d_extract_surprises_bernoth.py` | Bernoth-style residualized stance robustness | master and optional controls | augmented master and robustness summary | models, pandas, numpy | helper promoted; original archived |
| `run_main_pipeline.py` | Authoritative supplied modeling orchestrator | master dataset, optional daily stance files | final result CSVs and summary | models, forecast_oos, arch, scipy | archived and exposed via wrapper |
| `06_make_report_artifacts.py` | Create paper-ready tables and figures | final result CSVs and master dataset | LaTeX tables and PNG figures | models, matplotlib, pandas | archived; supplied outputs tracked |
| `model_data_master.csv` | Final processed modeling dataset | output of prior stages | input to all model stages | none | tracked under `data/processed/` |
| `Macroeconometrics.pdf` | Accompanying final paper | final empirical results | paper artifact | none | tracked under `paper/` |
| `Outputs/` | Final model tables, summary, figures, and compiled tables | generated modeling outputs | public research artifacts | none at read time | tracked under `reports/` |

## Execution Order

```text
01a + 01b
    └── 02
         └── 03a [03b only for mixed legacy/fresh scores]
              └── 04a [04b optional]
                   └── 05a
                        └── run_main_pipeline
                             └── 06

05d is an appendix robustness branch from the master dataset.
```

## Professional Cleanup Findings

- The supplied Python files use repository-relative paths; no hardcoded Mac
  user path was found.
- Coursework-oriented wording and historical handoff comments remain only in
  the traceability archive.
- The public paper copy was sanitized to remove student IDs and personal
  emails while preserving authorship and scientific content.
- Top-level script duplication and inconsistent model labels were resolved in
  the supported package and documented where the paper uses different IDs.
- Raw data, intermediate corpora, model caches, logs, and generated forecast
  paths are excluded from Git.
