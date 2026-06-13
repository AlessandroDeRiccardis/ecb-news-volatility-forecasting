from __future__ import annotations

from pathlib import Path

from ecb_vol_forecasting.config import load_config


def test_config_loading(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[research]\ninsample_end='2010-12-31'\n"
        "[forecasting]\nstep=10\nhorizons=[1]\nschemes=['RW']\n"
        "[runtime]\nrandom_seed=7\n"
    )
    config = load_config(path)
    assert config.insample_end == "2010-12-31"
    assert config.forecast_step == 10
    assert config.horizons == (1,)
    assert config.random_seed == 7
