# Legacy Archive

`original_scripts/` preserves the supplied implementation unchanged for
traceability. It is not the public API of this repository.

The scripts reconstruct the original numbered workflow and contain valuable
research logic, but they also use legacy `market_data/` and `output/` paths,
configure logging at import time, and expose inconsistent model labels across
entry points. One known issue is that `03b_merge_sources.py` imports
`ecb_sentiment_pipeline_v4`, a filename not present in the supplied snapshot.

Use the package under `src/ecb_vol_forecasting/` for reusable logic and the
top-level `scripts/` directory for supported commands.
