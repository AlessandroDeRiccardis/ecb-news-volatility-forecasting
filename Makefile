PYTHON ?= python

.PHONY: install test lint format data score models forecasts figures reproduce legacy-results quick clean

install:
	$(PYTHON) -m pip install -e ".[data,dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

data:
	$(PYTHON) scripts/build_data.py

score:
	$(PYTHON) scripts/score_stance.py --resume

models:
	$(PYTHON) scripts/estimate_models.py

forecasts:
	$(PYTHON) scripts/forecast_oos.py

figures:
	$(PYTHON) scripts/make_artifacts.py --skip-evaluation

reproduce:
	$(PYTHON) scripts/run_pipeline.py

legacy-results:
	$(PYTHON) scripts/run_legacy_results.py

quick:
	$(PYTHON) scripts/run_pipeline.py --quick

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov reports/forecasts
