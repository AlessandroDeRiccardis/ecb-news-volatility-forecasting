# Reproducibility Status

## Available Now

The tracked processed master dataset is sufficient to:

- validate the modeling schema;
- estimate the main model suite;
- regenerate rolling and increasing-window OOS forecasts;
- recompute the core forecast-accuracy table;
- regenerate descriptive figures.

The supplied final tables, figures, and consolidated summary are tracked under
`reports/`.

## Not Available In The Supplied Snapshot

- raw market and control CSVs;
- cleaned ECB document text;
- document master index;
- sentence-level FOMC-RoBERTa scores;
- document-level and calendar-day stance files;
- per-origin OOS forecast files.

Therefore, exact raw-to-paper replication is not possible from the snapshot
alone. Raw acquisition and scoring logic is preserved under `legacy/`, while
the processed-to-model workflow is supported by the refactored package.

## Commands

```bash
make install
make test
make data
make models
make forecasts
make figures
make reproduce
make legacy-results
```

For a short smoke run:

```bash
make quick
```

The full forecast suite is computationally expensive because every model is
re-estimated at each weekly OOS origin.

NA-GARCH likelihoods are non-convex. A verified refactored re-fit reached a
slightly better `B2.2` local optimum than the supplied final table. Exact
parameter-level replication therefore depends on optimizer path and restart
settings, although the model ranking and empirical conclusion are unchanged.
