# Data Guide

The repository includes one processed modeling snapshot:

- `processed/model_data_master.csv`: 4,201 Euro Stoxx 50 trading days from
  2007-04-02 through 2023-12-29, with returns, stance inputs, robustness
  variants, surprise inputs, controls, and fixed sample flags.

## Data Policy

Raw and intermediate inputs are intentionally excluded from Git. The missing
inputs are public or pipeline-generated, but are either large, expensive to
rebuild, or not present in the supplied project snapshot.

| Layer | Expected contents | Git policy |
|---|---|---|
| `raw/` | Euro Stoxx 50 prices, VSTOXX, ECB yield curves, ECB documents | ignored |
| `external/` | EUR/USD, VIX, Brent and other downloaded controls | ignored |
| `interim/` | cleaned ECB text, sentence scores, document and daily stance | ignored |
| `processed/` | compact master modeling dataset | tracked |

## Required Schemas

Sentence scores require:

`doc_id,pub_date,doc_type,speaker,sentence_idx,sentence,dovish_score,hawkish_score,neutral_score`

The processed master requires:

`date,period,log_return,sq_return,abs_return,sq_return_5d,P_t,N_t,S_t`

Additional tracked columns support source, threshold, surprise, and control
robustness checks. Run `make data` to validate the processed snapshot.

## Rebuilding From Raw Inputs

The archived collection and scoring code is retained in `legacy/original_scripts/`.
It documents the full raw-to-master workflow, but the supplied snapshot lacks
cleaned ECB documents and sentence-level model scores. Rebuilding those stages
requires network access, the `gtfintechlab/FOMC-RoBERTa` model, approximately
seven CPU-hours for stance scoring, and manual review of acquired data.

Do not treat Yahoo Finance or ECB downloads as immutable. Record retrieval
dates and checksums before claiming exact replication.
