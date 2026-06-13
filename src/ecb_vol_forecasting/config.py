"""Project configuration and repository-relative paths."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Repository paths used by the reproducible pipeline."""

    root: Path = PROJECT_ROOT

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def data_external(self) -> Path:
        return self.root / "data" / "external"

    @property
    def master_dataset(self) -> Path:
        return self.data_processed / "model_data_master.csv"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def tables(self) -> Path:
        return self.reports / "tables"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def forecasts(self) -> Path:
        return self.reports / "forecasts"

    @property
    def legacy_scripts(self) -> Path:
        return self.root / "legacy" / "original_scripts"

    def ensure_output_dirs(self) -> None:
        """Create generated-output directories."""
        for path in (self.tables, self.figures, self.forecasts):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ProjectConfig:
    """Research design settings shared across scripts."""

    presample_start: str = "2007-04-02"
    insample_start: str = "2008-01-01"
    insample_end: str = "2018-12-31"
    outsample_end: str = "2023-12-31"
    forecast_step: int = 5
    horizons: tuple[int, ...] = (1, 5)
    schemes: tuple[str, ...] = ("RW", "IW")
    distribution: str = "studentst"
    random_seed: int = 42
    paths: ProjectPaths = field(default_factory=ProjectPaths)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load a TOML configuration file, falling back to documented defaults."""
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.toml"
    if not config_path.exists():
        return ProjectConfig()

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    research = raw.get("research", {})
    forecasting = raw.get("forecasting", {})
    runtime = raw.get("runtime", {})
    return ProjectConfig(
        presample_start=research.get("presample_start", "2007-04-02"),
        insample_start=research.get("insample_start", "2008-01-01"),
        insample_end=research.get("insample_end", "2018-12-31"),
        outsample_end=research.get("outsample_end", "2023-12-31"),
        forecast_step=int(forecasting.get("step", 5)),
        horizons=tuple(int(x) for x in forecasting.get("horizons", [1, 5])),
        schemes=tuple(forecasting.get("schemes", ["RW", "IW"])),
        distribution=forecasting.get("distribution", "studentst"),
        random_seed=int(runtime.get("random_seed", 42)),
    )
