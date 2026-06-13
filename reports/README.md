# Reports

This folder contains the supplied final empirical outputs.

- `tables/`: machine-readable final result tables.
- `figures/`: paper-ready figures.
- `compiled/`: supplied LaTeX tables and compiled PDF.
- `final_results_summary.txt`: consolidated model and evaluation output.

The aggregate results are sufficient to audit the headline findings. The
per-origin OOS forecast CSVs were not supplied, so DM tests and the Model
Confidence Set cannot be recalculated directly from these tracked outputs.
Run `make forecasts` to regenerate forecast paths from the processed master.
